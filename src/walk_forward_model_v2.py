"""
V2 walk-forward factor model.

The model contains five economically distinct components:

1. Turnover stability
2. Low risk
3. 12-1 momentum
4. Value
5. Short-term reversal

For every formation month, factor weights are estimated using only
IC observations whose target return date is already observable.

This avoids using future returns when creating historical rankings.
"""

from __future__ import annotations

import math
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from src.factors_v2 import resolve_database_path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "reports" / "results"

FACTOR_TABLE = "processed_factor_values_v2"
RETURN_TABLE = "factor_forward_returns_v2"

COMPONENT_IC_TABLE = "walk_forward_component_ic_v2"
WEIGHT_TABLE = "walk_forward_factor_weights_v2"
SCORE_TABLE = "walk_forward_stock_scores_v2"
WEIGHT_SUMMARY_TABLE = "walk_forward_weight_summary_v2"

COMPONENT_IC_FILE = (
    RESULTS_DIR / "walk_forward_component_ic_v2.csv"
)
WEIGHT_FILE = (
    RESULTS_DIR / "walk_forward_factor_weights_v2.csv"
)
SCORE_FILE = (
    RESULTS_DIR / "walk_forward_stock_scores_v2.csv"
)
WEIGHT_SUMMARY_FILE = (
    RESULTS_DIR / "walk_forward_weight_summary_v2.csv"
)

TRAINING_WINDOW_MONTHS = 36
MINIMUM_TRAINING_MONTHS = 24
IC_HALF_LIFE_MONTHS = 12
MAXIMUM_COMPONENT_WEIGHT = 0.35
WEIGHT_UPDATE_RATE = 0.30
MINIMUM_IC_STOCKS = 30

COMPONENTS = {
    "turnover_stability_component": {
        "component_name": "Turnover Stability",
        "input_columns": [
            "turnover_stability_20_neutral",
        ],
    },
    "low_risk_component": {
        "component_name": "Low Risk",
        "input_columns": [
            "volatility_20_neutral",
            "downside_volatility_60_neutral",
        ],
    },
    "long_momentum_component": {
        "component_name": "12-1 Momentum",
        "input_columns": [
            "momentum_12_1_neutral",
        ],
    },
    "value_component": {
        "component_name": "Value",
        "input_columns": [
            "value_pe_score_neutral",
            "value_pb_score_neutral",
        ],
    },
    "short_reversal_component": {
        "component_name": "Short Reversal",
        "input_columns": [
            "reversal_5_score_neutral",
        ],
    },
}

IDENTIFIER_COLUMNS = [
    "snapshot_date",
    "signal_price_date",
    "baostock_code",
    "symbol",
    "stock_name",
    "industry",
    "close",
]


def table_exists(
    connection: sqlite3.Connection,
    table_name: str,
) -> bool:
    result = connection.execute(
        """
        SELECT COUNT(*)
        FROM sqlite_master
        WHERE type = 'table'
          AND name = ?;
        """,
        (table_name,),
    ).fetchone()

    return bool(result and result[0] > 0)


def zscore(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(
        values,
        errors="coerce",
    ).astype(float)

    mean_value = values.mean()
    standard_deviation = values.std(ddof=0)

    if (
        pd.isna(standard_deviation)
        or standard_deviation < 1e-12
    ):
        return pd.Series(
            0.0,
            index=values.index,
            dtype=float,
        )

    return (
        values - mean_value
    ) / standard_deviation


def spearman_ic(
    scores: pd.Series,
    future_returns: pd.Series,
) -> float:
    valid = pd.DataFrame(
        {
            "score": scores,
            "future_return": future_returns,
        }
    ).dropna()

    if len(valid) < MINIMUM_IC_STOCKS:
        return np.nan

    if (
        valid["score"].nunique() < 3
        or valid["future_return"].nunique() < 3
    ):
        return np.nan

    score_rank = valid["score"].rank(
        method="average"
    )
    return_rank = valid[
        "future_return"
    ].rank(method="average")

    return float(
        score_rank.corr(return_rank)
    )


def load_model_panel(
    connection: sqlite3.Connection,
) -> pd.DataFrame:
    factors = pd.read_sql_query(
        f'SELECT * FROM "{FACTOR_TABLE}"',
        connection,
    )

    forward_returns = pd.read_sql_query(
        f"""
        SELECT
            snapshot_date,
            baostock_code,
            target_snapshot_date_1m,
            forward_return_1m
        FROM "{RETURN_TABLE}";
        """,
        connection,
    )

    required_input_columns = {
        column
        for component in COMPONENTS.values()
        for column in component["input_columns"]
    }

    required_factor_columns = {
        "snapshot_date",
        "signal_price_date",
        "baostock_code",
        "close",
        *required_input_columns,
    }

    missing_factor_columns = (
        required_factor_columns.difference(
            factors.columns
        )
    )

    if missing_factor_columns:
        raise ValueError(
            f"{FACTOR_TABLE} is missing columns: "
            f"{sorted(missing_factor_columns)}"
        )

    for column in [
        "snapshot_date",
        "signal_price_date",
    ]:
        factors[column] = pd.to_datetime(
            factors[column],
            errors="coerce",
        )

    forward_returns[
        "snapshot_date"
    ] = pd.to_datetime(
        forward_returns["snapshot_date"],
        errors="coerce",
    )

    forward_returns[
        "target_snapshot_date_1m"
    ] = pd.to_datetime(
        forward_returns[
            "target_snapshot_date_1m"
        ],
        errors="coerce",
    )

    forward_returns[
        "forward_return_1m"
    ] = pd.to_numeric(
        forward_returns[
            "forward_return_1m"
        ],
        errors="coerce",
    )

    numeric_factor_columns = [
        "close",
        *required_input_columns,
    ]

    for column in numeric_factor_columns:
        factors[column] = pd.to_numeric(
            factors[column],
            errors="coerce",
        )

    factors[numeric_factor_columns] = factors[
        numeric_factor_columns
    ].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    panel = factors.merge(
        forward_returns,
        on=[
            "snapshot_date",
            "baostock_code",
        ],
        how="left",
        validate="one_to_one",
    )

    panel = panel.dropna(
        subset=[
            "snapshot_date",
            "baostock_code",
        ]
    )

    panel = panel.sort_values(
        ["snapshot_date", "baostock_code"]
    ).reset_index(drop=True)

    return panel


def build_component_scores(
    panel: pd.DataFrame,
) -> pd.DataFrame:
    processed_snapshots = []

    for snapshot_date, snapshot in panel.groupby(
        "snapshot_date",
        sort=True,
    ):
        snapshot = snapshot.copy()

        for (
            component_column,
            component_information,
        ) in COMPONENTS.items():
            input_columns = (
                component_information[
                    "input_columns"
                ]
            )

            raw_component = snapshot[
                input_columns
            ].mean(axis=1)

            snapshot[component_column] = zscore(
                raw_component
            )

        component_columns = list(
            COMPONENTS.keys()
        )

        snapshot[
            "fixed_equal_weight_score"
        ] = zscore(
            snapshot[
                component_columns
            ].mean(axis=1)
        )

        processed_snapshots.append(snapshot)

    return pd.concat(
        processed_snapshots,
        ignore_index=True,
    )


def calculate_component_ic(
    panel: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for snapshot_date, snapshot in panel.groupby(
        "snapshot_date",
        sort=True,
    ):
        target_dates = snapshot[
            "target_snapshot_date_1m"
        ].dropna()

        if target_dates.empty:
            continue

        target_snapshot_date = (
            target_dates.iloc[0]
        )

        for (
            component_column,
            component_information,
        ) in COMPONENTS.items():
            valid = snapshot[
                [
                    component_column,
                    "forward_return_1m",
                ]
            ].dropna()

            ic_value = spearman_ic(
                scores=valid[component_column],
                future_returns=valid[
                    "forward_return_1m"
                ],
            )

            if pd.isna(ic_value):
                continue

            rows.append(
                {
                    "snapshot_date": snapshot_date,
                    "target_snapshot_date": (
                        target_snapshot_date
                    ),
                    "component_column": (
                        component_column
                    ),
                    "component_name": (
                        component_information[
                            "component_name"
                        ]
                    ),
                    "number_of_stocks": len(valid),
                    "rank_ic": ic_value,
                }
            )

    return pd.DataFrame(rows).sort_values(
        [
            "target_snapshot_date",
            "component_column",
        ]
    ).reset_index(drop=True)


def exponentially_weighted_ic_statistics(
    history: pd.DataFrame,
) -> dict:
    history = history.sort_values(
        "target_snapshot_date"
    ).tail(TRAINING_WINDOW_MONTHS)

    ic_values = history[
        "rank_ic"
    ].to_numpy(dtype=float)

    number_of_observations = len(ic_values)

    if number_of_observations == 0:
        return {
            "number_of_training_periods": 0,
            "training_start_date": pd.NaT,
            "training_end_date": pd.NaT,
            "weighted_mean_ic": np.nan,
            "weighted_ic_std": np.nan,
            "positive_ic_rate_percent": np.nan,
            "raw_strength": 0.0,
        }

    ages = np.arange(
        number_of_observations - 1,
        -1,
        -1,
        dtype=float,
    )

    exponential_weights = np.power(
        0.5,
        ages / IC_HALF_LIFE_MONTHS,
    )

    exponential_weights = (
        exponential_weights
        / exponential_weights.sum()
    )

    weighted_mean_ic = float(
        np.sum(
            exponential_weights
            * ic_values
        )
    )

    weighted_variance = float(
        np.sum(
            exponential_weights
            * (
                ic_values
                - weighted_mean_ic
            )
            ** 2
        )
    )

    weighted_ic_std = math.sqrt(
        max(weighted_variance, 0.0)
    )

    positive_rate = float(
        (ic_values > 0).mean() * 100
    )

    if (
        number_of_observations
        >= MINIMUM_TRAINING_MONTHS
        and weighted_mean_ic > 0
    ):
        raw_strength = (
            weighted_mean_ic
            / max(weighted_ic_std, 0.05)
        )
    else:
        raw_strength = 0.0

    return {
        "number_of_training_periods": (
            number_of_observations
        ),
        "training_start_date": history[
            "target_snapshot_date"
        ].min(),
        "training_end_date": history[
            "target_snapshot_date"
        ].max(),
        "weighted_mean_ic": (
            weighted_mean_ic
        ),
        "weighted_ic_std": (
            weighted_ic_std
        ),
        "positive_ic_rate_percent": (
            positive_rate
        ),
        "raw_strength": (
            raw_strength
        ),
    }


def cap_component_weights(
    raw_strengths: pd.Series,
) -> pd.Series:
    strengths = raw_strengths.clip(
        lower=0.0
    ).astype(float)

    result = pd.Series(
        0.0,
        index=strengths.index,
        dtype=float,
    )

    active = strengths.loc[
        strengths > 0
    ].copy()

    if active.empty:
        return result

    remaining_weight = 1.0

    while not active.empty:
        proposed = (
            active
            / active.sum()
            * remaining_weight
        )

        over_cap = (
            proposed
            > MAXIMUM_COMPONENT_WEIGHT
        )

        if not over_cap.any():
            result.loc[
                proposed.index
            ] = proposed
            break

        capped_names = proposed.loc[
            over_cap
        ].index

        result.loc[
            capped_names
        ] = MAXIMUM_COMPONENT_WEIGHT

        remaining_weight -= (
            len(capped_names)
            * MAXIMUM_COMPONENT_WEIGHT
        )

        active = active.drop(
            index=capped_names
        )

        if remaining_weight <= 1e-12:
            break

    return result


def build_walk_forward_model(
    panel: pd.DataFrame,
    component_ic: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    score_snapshots = []
    weight_rows = []

    previous_weights: pd.Series | None = None

    snapshot_dates = sorted(
        panel["snapshot_date"].unique()
    )

    component_columns = list(
        COMPONENTS.keys()
    )

    for position, snapshot_date in enumerate(
        snapshot_dates,
        start=1,
    ):
        snapshot = panel.loc[
            panel["snapshot_date"]
            == snapshot_date
        ].copy()

        component_statistics = {}
        raw_strengths = {}

        for component_column in component_columns:
            history = component_ic.loc[
                (
                    component_ic[
                        "component_column"
                    ]
                    == component_column
                )
                & (
                    component_ic[
                        "target_snapshot_date"
                    ]
                    <= snapshot_date
                )
            ].copy()

            statistics = (
                exponentially_weighted_ic_statistics(
                    history
                )
            )

            component_statistics[
                component_column
            ] = statistics

            raw_strengths[
                component_column
            ] = statistics[
                "raw_strength"
            ]

        enough_history = all(
            component_statistics[column][
                "number_of_training_periods"
            ]
            >= MINIMUM_TRAINING_MONTHS
            for column in component_columns
        )

        target_weights = pd.Series(
            0.0,
            index=component_columns,
            dtype=float,
        )

        model_ready = False
        weight_method = "INSUFFICIENT_HISTORY"

        if enough_history:
            target_weights = (
                cap_component_weights(
                    pd.Series(raw_strengths)
                )
            )

            if target_weights.sum() > 0:
                model_ready = True
                weight_method = (
                    "ROLLING_IC_WEIGHT"
                )

            elif previous_weights is not None:
                target_weights = (
                    previous_weights.copy()
                )
                model_ready = True
                weight_method = (
                    "PREVIOUS_WEIGHT_FALLBACK"
                )

        if model_ready:
            if previous_weights is None:
                smoothed_weights = (
                    target_weights.copy()
                )
            else:
                smoothed_weights = (
                    (
                        1.0
                        - WEIGHT_UPDATE_RATE
                    )
                    * previous_weights
                    + WEIGHT_UPDATE_RATE
                    * target_weights
                )

            previous_weights = (
                smoothed_weights.copy()
            )

            weighted_score = pd.Series(
                0.0,
                index=snapshot.index,
                dtype=float,
            )

            for component_column in (
                component_columns
            ):
                weighted_score += (
                    snapshot[component_column]
                    * smoothed_weights[
                        component_column
                    ]
                )

            snapshot[
                "walk_forward_score"
            ] = zscore(weighted_score)

            snapshot[
                "walk_forward_rank"
            ] = snapshot[
                "walk_forward_score"
            ].rank(
                method="first",
                ascending=False,
            ).astype("Int64")

        else:
            smoothed_weights = pd.Series(
                0.0,
                index=component_columns,
                dtype=float,
            )

            snapshot[
                "walk_forward_score"
            ] = np.nan

            snapshot[
                "walk_forward_rank"
            ] = pd.Series(
                pd.NA,
                index=snapshot.index,
                dtype="Int64",
            )

        snapshot[
            "fixed_equal_weight_rank"
        ] = snapshot[
            "fixed_equal_weight_score"
        ].rank(
            method="first",
            ascending=False,
        ).astype("Int64")

        snapshot["model_ready"] = int(
            model_ready
        )

        snapshot[
            "active_component_weight_sum"
        ] = smoothed_weights.sum()

        score_snapshots.append(snapshot)

        for component_column in component_columns:
            component_information = (
                COMPONENTS[component_column]
            )

            statistics = (
                component_statistics[
                    component_column
                ]
            )

            training_end_date = statistics[
                "training_end_date"
            ]

            lookahead_violation = int(
                pd.notna(training_end_date)
                and training_end_date
                > snapshot_date
            )

            weight_rows.append(
                {
                    "snapshot_date": snapshot_date,
                    "component_column": (
                        component_column
                    ),
                    "component_name": (
                        component_information[
                            "component_name"
                        ]
                    ),
                    "model_ready": int(
                        model_ready
                    ),
                    "weight_method": (
                        weight_method
                    ),
                    **statistics,
                    "target_weight": (
                        target_weights[
                            component_column
                        ]
                    ),
                    "smoothed_weight": (
                        smoothed_weights[
                            component_column
                        ]
                    ),
                    "lookahead_violation": (
                        lookahead_violation
                    ),
                }
            )

        if (
            position == 1
            or position % 10 == 0
            or position == len(snapshot_dates)
        ):
            print(
                f"[{position}/"
                f"{len(snapshot_dates)}] "
                f"Built model for "
                f"{pd.Timestamp(snapshot_date).date()}",
                flush=True,
            )

    scores = pd.concat(
        score_snapshots,
        ignore_index=True,
    )

    weights = pd.DataFrame(
        weight_rows
    )

    return scores, weights


def build_weight_summary(
    weights: pd.DataFrame,
) -> pd.DataFrame:
    ready_weights = weights.loc[
        weights["model_ready"] == 1
    ].copy()

    if ready_weights.empty:
        return pd.DataFrame()

    latest_snapshot = ready_weights[
        "snapshot_date"
    ].max()

    latest_weights = (
        ready_weights.loc[
            ready_weights["snapshot_date"]
            == latest_snapshot,
            [
                "component_column",
                "smoothed_weight",
            ],
        ]
        .set_index("component_column")[
            "smoothed_weight"
        ]
    )

    summary = (
        ready_weights.groupby(
            [
                "component_column",
                "component_name",
            ],
            as_index=False,
        )["smoothed_weight"]
        .agg(
            average_weight="mean",
            minimum_weight="min",
            maximum_weight="max",
        )
    )

    summary["latest_weight"] = summary[
        "component_column"
    ].map(latest_weights)

    summary[
        "latest_snapshot_date"
    ] = latest_snapshot

    return summary.sort_values(
        "latest_weight",
        ascending=False,
    ).reset_index(drop=True)


def select_score_columns(
    scores: pd.DataFrame,
) -> pd.DataFrame:
    desired_columns = [
        *IDENTIFIER_COLUMNS,
        "target_snapshot_date_1m",
        "forward_return_1m",
        *COMPONENTS.keys(),
        "fixed_equal_weight_score",
        "fixed_equal_weight_rank",
        "walk_forward_score",
        "walk_forward_rank",
        "model_ready",
        "active_component_weight_sum",
    ]

    return scores[
        [
            column
            for column in desired_columns
            if column in scores.columns
        ]
    ].sort_values(
        ["snapshot_date", "baostock_code"]
    ).reset_index(drop=True)


def format_dates(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    output = frame.copy()

    for column in output.columns:
        if (
            column.endswith("_date")
            or column == "snapshot_date"
        ):
            output[column] = pd.to_datetime(
                output[column],
                errors="coerce",
            ).dt.strftime("%Y-%m-%d")

    return output


def save_outputs(
    connection: sqlite3.Connection,
    component_ic: pd.DataFrame,
    weights: pd.DataFrame,
    scores: pd.DataFrame,
    weight_summary: pd.DataFrame,
) -> None:
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_items = [
        (
            COMPONENT_IC_TABLE,
            format_dates(component_ic),
            COMPONENT_IC_FILE,
        ),
        (
            WEIGHT_TABLE,
            format_dates(weights),
            WEIGHT_FILE,
        ),
        (
            SCORE_TABLE,
            format_dates(scores),
            SCORE_FILE,
        ),
        (
            WEIGHT_SUMMARY_TABLE,
            format_dates(weight_summary),
            WEIGHT_SUMMARY_FILE,
        ),
    ]

    for (
        table_name,
        frame,
        csv_path,
    ) in output_items:
        frame.to_sql(
            table_name,
            connection,
            if_exists="replace",
            index=False,
            chunksize=2_000,
        )

        frame.to_csv(
            csv_path,
            index=False,
            encoding="utf-8-sig",
            float_format="%.10f",
        )

    connection.execute(
        f"""
        CREATE INDEX IF NOT EXISTS
        idx_{SCORE_TABLE}_snapshot_rank
        ON {SCORE_TABLE} (
            snapshot_date,
            walk_forward_rank
        );
        """
    )

    connection.commit()


def print_latest_weights(
    weights: pd.DataFrame,
) -> None:
    ready = weights.loc[
        weights["model_ready"] == 1
    ].copy()

    if ready.empty:
        print(
            "No walk-forward model became ready."
        )
        return

    latest_snapshot = ready[
        "snapshot_date"
    ].max()

    latest = ready.loc[
        ready["snapshot_date"]
        == latest_snapshot
    ].sort_values(
        "smoothed_weight",
        ascending=False,
    )

    display_columns = [
        "component_name",
        "number_of_training_periods",
        "weighted_mean_ic",
        "weighted_ic_std",
        "positive_ic_rate_percent",
        "target_weight",
        "smoothed_weight",
    ]

    print()
    print("=" * 120)
    print(
        f"LATEST WALK-FORWARD WEIGHTS: "
        f"{latest_snapshot.date()}"
    )
    print("=" * 120)

    print(
        latest[display_columns].to_string(
            index=False,
            formatters={
                "weighted_mean_ic": (
                    lambda value: f"{value:.4f}"
                ),
                "weighted_ic_std": (
                    lambda value: f"{value:.4f}"
                ),
                "positive_ic_rate_percent": (
                    lambda value: f"{value:.2f}%"
                ),
                "target_weight": (
                    lambda value: f"{value:.2%}"
                ),
                "smoothed_weight": (
                    lambda value: f"{value:.2%}"
                ),
            },
        )
    )


def print_latest_ranking(
    scores: pd.DataFrame,
) -> None:
    ready_scores = scores.loc[
        scores["model_ready"] == 1
    ].copy()

    if ready_scores.empty:
        return

    latest_snapshot = ready_scores[
        "snapshot_date"
    ].max()

    latest = ready_scores.loc[
        ready_scores["snapshot_date"]
        == latest_snapshot
    ].sort_values(
        "walk_forward_rank"
    ).head(20)

    display_columns = [
        "walk_forward_rank",
        "symbol",
        "stock_name",
        "industry",
        "walk_forward_score",
    ]

    display_columns = [
        column
        for column in display_columns
        if column in latest.columns
    ]

    print()
    print("=" * 120)
    print(
        f"LATEST WALK-FORWARD TOP 20: "
        f"{latest_snapshot.date()}"
    )
    print("=" * 120)

    print(
        latest[display_columns].to_string(
            index=False,
            formatters={
                "walk_forward_score": (
                    lambda value: f"{value:.4f}"
                )
            },
        )
    )


def main() -> None:
    pd.set_option(
        "display.max_columns",
        None,
    )
    pd.set_option(
        "display.width",
        220,
    )

    database_path = resolve_database_path()

    print("=" * 120)
    print("V2 WALK-FORWARD FACTOR MODEL")
    print("=" * 120)
    print(f"Database: {database_path}")
    print(
        f"Training window: "
        f"{TRAINING_WINDOW_MONTHS} months"
    )
    print(
        f"Minimum training history: "
        f"{MINIMUM_TRAINING_MONTHS} months"
    )
    print(
        f"IC half-life: "
        f"{IC_HALF_LIFE_MONTHS} months"
    )
    print(
        f"Maximum component weight: "
        f"{MAXIMUM_COMPONENT_WEIGHT:.0%}"
    )
    print()

    with sqlite3.connect(
        database_path
    ) as connection:
        for table_name in [
            FACTOR_TABLE,
            RETURN_TABLE,
        ]:
            if not table_exists(
                connection,
                table_name,
            ):
                raise RuntimeError(
                    f"Required table does not exist: "
                    f"{table_name}"
                )

        print(
            "Loading factor and return data...",
            flush=True,
        )

        panel = load_model_panel(connection)

        print(
            f"Observations: {len(panel):,}"
        )
        print(
            f"Snapshots: "
            f"{panel['snapshot_date'].nunique():,}"
        )

        print(
            "Building candidate factor components...",
            flush=True,
        )

        panel = build_component_scores(panel)

        print(
            "Calculating historical component IC...",
            flush=True,
        )

        component_ic = (
            calculate_component_ic(panel)
        )

        print(
            "Building walk-forward weights...",
            flush=True,
        )

        scores, weights = (
            build_walk_forward_model(
                panel=panel,
                component_ic=component_ic,
            )
        )

        scores = select_score_columns(scores)

        weight_summary = (
            build_weight_summary(weights)
        )

        lookahead_violations = int(
            weights[
                "lookahead_violation"
            ].sum()
        )

        ready_dates = scores.loc[
            scores["model_ready"] == 1,
            "snapshot_date",
        ].drop_duplicates()

        print()
        print("=" * 120)
        print("WALK-FORWARD MODEL SUMMARY")
        print("=" * 120)
        print(
            f"Component IC observations: "
            f"{len(component_ic):,}"
        )
        print(
            f"Model-ready snapshots: "
            f"{len(ready_dates):,}"
        )

        if not ready_dates.empty:
            print(
                f"First model-ready snapshot: "
                f"{ready_dates.min().date()}"
            )
            print(
                f"Latest model-ready snapshot: "
                f"{ready_dates.max().date()}"
            )

        print(
            f"Look-ahead violations: "
            f"{lookahead_violations}"
        )

        if lookahead_violations:
            raise RuntimeError(
                "Look-ahead audit failed."
            )

        print(
            "Saving walk-forward model...",
            flush=True,
        )

        save_outputs(
            connection=connection,
            component_ic=component_ic,
            weights=weights,
            scores=scores,
            weight_summary=weight_summary,
        )

    print_latest_weights(weights)
    print_latest_ranking(scores)

    print()
    print("=" * 120)
    print("OUTPUT FILES")
    print("=" * 120)
    print(COMPONENT_IC_FILE)
    print(WEIGHT_FILE)
    print(SCORE_FILE)
    print(WEIGHT_SUMMARY_FILE)
    print()
    print(
        "V2 walk-forward factor model completed."
    )


if __name__ == "__main__":
    main()