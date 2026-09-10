"""
V2 factor evaluation.

This module evaluates processed monthly factors using:

- Forward 1-month returns
- Forward 3-month returns
- Monthly Spearman Rank IC
- IC mean, standard deviation, ICIR and t-statistic
- Positive IC rate
- Five-quantile portfolio returns
- Q5 minus Q1 spread
- Quantile monotonicity
- Standardized versus neutralized factor comparison
- Factor IC decay

Factor values are measured at the signal date.
Future returns are calculated only from later market prices.
"""

from __future__ import annotations

import math
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from src.factor_processing_v2 import (
    ALPHA_METADATA,
    CATEGORY_DEFINITIONS,
)
from src.factors_v2 import resolve_database_path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "reports" / "results"

FACTOR_TABLE = "processed_factor_values_v2"
PRICE_TABLE = "daily_prices_v2"

FORWARD_RETURN_TABLE = "factor_forward_returns_v2"
MONTHLY_IC_TABLE = "monthly_factor_ic_v2"
IC_SUMMARY_TABLE = "factor_ic_summary_v2"
MONTHLY_QUANTILE_TABLE = "monthly_factor_quantile_returns_v2"
QUANTILE_SUMMARY_TABLE = "factor_quantile_summary_v2"
FACTOR_DECAY_TABLE = "factor_decay_summary_v2"

FORWARD_RETURN_FILE = (
    RESULTS_DIR / "factor_forward_returns_v2.csv"
)
MONTHLY_IC_FILE = (
    RESULTS_DIR / "monthly_factor_ic_v2.csv"
)
IC_SUMMARY_FILE = (
    RESULTS_DIR / "factor_ic_summary_v2.csv"
)
MONTHLY_QUANTILE_FILE = (
    RESULTS_DIR / "monthly_factor_quantile_returns_v2.csv"
)
QUANTILE_SUMMARY_FILE = (
    RESULTS_DIR / "factor_quantile_summary_v2.csv"
)
FACTOR_DECAY_FILE = (
    RESULTS_DIR / "factor_decay_summary_v2.csv"
)

HORIZONS = {
    1: "forward_return_1m",
    3: "forward_return_3m",
}

MINIMUM_IC_STOCKS = 30
MINIMUM_QUANTILE_STOCKS = 50
MAXIMUM_TARGET_PRICE_STALENESS_DAYS = 14


CATEGORY_NAMES = {
    "momentum_category_score": "Momentum Category",
    "reversal_category_score": "Reversal Category",
    "risk_category_score": "Risk Category",
    "trend_category_score": "Trend Category",
    "liquidity_category_score": "Liquidity Category",
    "value_category_score": "Value Category",
}


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


def build_signal_metadata() -> pd.DataFrame:
    rows: list[dict] = []

    for item in ALPHA_METADATA:
        base_factor = item["factor_column"]

        rows.append(
            {
                "signal_column": (
                    f"{base_factor}_zscore"
                ),
                "base_factor": base_factor,
                "factor_name": item["factor_name"],
                "category": item["category"],
                "signal_variant": "standardized",
            }
        )

        rows.append(
            {
                "signal_column": (
                    f"{base_factor}_neutral"
                ),
                "base_factor": base_factor,
                "factor_name": item["factor_name"],
                "category": item["category"],
                "signal_variant": "neutralized",
            }
        )

    for category_column in CATEGORY_DEFINITIONS:
        rows.append(
            {
                "signal_column": category_column,
                "base_factor": category_column,
                "factor_name": CATEGORY_NAMES.get(
                    category_column,
                    category_column,
                ),
                "category": "Category Composite",
                "signal_variant": "category",
            }
        )

    rows.append(
        {
            "signal_column": (
                "equal_weight_composite_score"
            ),
            "base_factor": (
                "equal_weight_composite_score"
            ),
            "factor_name": (
                "Equal-Weight Composite"
            ),
            "category": "Composite",
            "signal_variant": "composite",
        }
    )

    return pd.DataFrame(rows)


def load_processed_factors(
    connection: sqlite3.Connection,
    signal_metadata: pd.DataFrame,
) -> pd.DataFrame:
    factors = pd.read_sql_query(
        f'SELECT * FROM "{FACTOR_TABLE}"',
        connection,
    )

    signal_columns = signal_metadata[
        "signal_column"
    ].tolist()

    required_columns = {
        "snapshot_date",
        "signal_price_date",
        "baostock_code",
        "close",
        *signal_columns,
    }

    missing_columns = required_columns.difference(
        factors.columns
    )

    if missing_columns:
        raise ValueError(
            f"{FACTOR_TABLE} is missing columns: "
            f"{sorted(missing_columns)}"
        )

    factors["snapshot_date"] = pd.to_datetime(
        factors["snapshot_date"],
        errors="coerce",
    )

    factors["signal_price_date"] = pd.to_datetime(
        factors["signal_price_date"],
        errors="coerce",
    )

    numeric_columns = [
        "close",
        *signal_columns,
    ]

    for column in numeric_columns:
        factors[column] = pd.to_numeric(
            factors[column],
            errors="coerce",
        )

    factors[numeric_columns] = factors[
        numeric_columns
    ].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    factors = factors.dropna(
        subset=[
            "snapshot_date",
            "signal_price_date",
            "baostock_code",
            "close",
        ]
    )

    factors = factors.loc[
        factors["close"] > 0
    ].copy()

    duplicate_count = int(
        factors.duplicated(
            ["snapshot_date", "baostock_code"]
        ).sum()
    )

    if duplicate_count:
        raise ValueError(
            f"{FACTOR_TABLE} contains "
            f"{duplicate_count:,} duplicate rows."
        )

    return factors.sort_values(
        ["snapshot_date", "baostock_code"]
    ).reset_index(drop=True)


def load_daily_prices(
    connection: sqlite3.Connection,
) -> pd.DataFrame:
    prices = pd.read_sql_query(
        f"""
        SELECT
            date,
            baostock_code,
            close,
            trade_status
        FROM {PRICE_TABLE};
        """,
        connection,
    )

    prices["date"] = pd.to_datetime(
        prices["date"],
        errors="coerce",
    )

    prices["close"] = pd.to_numeric(
        prices["close"],
        errors="coerce",
    )

    prices["trade_status"] = pd.to_numeric(
        prices["trade_status"],
        errors="coerce",
    )

    prices = prices.dropna(
        subset=[
            "date",
            "baostock_code",
            "close",
        ]
    )

    prices = prices.loc[
        (prices["trade_status"] == 1)
        & (prices["close"] > 0)
    ].copy()

    prices = prices.sort_values(
        ["baostock_code", "date"]
    )

    prices = prices.drop_duplicates(
        ["baostock_code", "date"],
        keep="last",
    )

    return prices.reset_index(drop=True)


def add_target_snapshot_dates(
    factor_panel: pd.DataFrame,
) -> pd.DataFrame:
    panel = factor_panel.copy()

    snapshot_dates = sorted(
        panel["snapshot_date"]
        .dropna()
        .unique()
    )

    for horizon_months in HORIZONS:
        target_mapping = {}

        for position, snapshot_date in enumerate(
            snapshot_dates
        ):
            target_position = (
                position + horizon_months
            )

            if target_position < len(
                snapshot_dates
            ):
                target_mapping[
                    snapshot_date
                ] = snapshot_dates[
                    target_position
                ]

        target_column = (
            f"target_snapshot_date_"
            f"{horizon_months}m"
        )

        panel[target_column] = panel[
            "snapshot_date"
        ].map(target_mapping)

    return panel


def attach_one_horizon_prices(
    panel: pd.DataFrame,
    prices: pd.DataFrame,
    horizon_months: int,
) -> pd.DataFrame:
    target_snapshot_column = (
        f"target_snapshot_date_{horizon_months}m"
    )
    target_price_date_column = (
        f"target_price_date_{horizon_months}m"
    )
    target_close_column = (
        f"target_close_{horizon_months}m"
    )
    return_column = HORIZONS[
        horizon_months
    ]

    panel[target_price_date_column] = pd.NaT
    panel[target_close_column] = np.nan

    price_groups = {
        code: group[
            ["date", "close"]
        ].sort_values("date")
        for code, group in prices.groupby(
            "baostock_code",
            sort=False,
        )
    }

    grouped_indices = panel.groupby(
        "baostock_code",
        sort=False,
    ).groups

    for baostock_code, row_indices in (
        grouped_indices.items()
    ):
        stock_prices = price_groups.get(
            baostock_code
        )

        if stock_prices is None:
            continue

        requests = panel.loc[
            row_indices,
            [
                target_snapshot_column,
                "signal_price_date",
            ],
        ].copy()

        requests = requests.dropna(
            subset=[target_snapshot_column]
        )

        if requests.empty:
            continue

        requests["_panel_index"] = (
            requests.index
        )

        requests = requests.sort_values(
            target_snapshot_column
        )

        lookup_prices = stock_prices.rename(
            columns={
                "date": target_price_date_column,
                "close": target_close_column,
            }
        )

        matched = pd.merge_asof(
            requests,
            lookup_prices,
            left_on=target_snapshot_column,
            right_on=target_price_date_column,
            direction="backward",
            allow_exact_matches=True,
        )

        stale_days = (
            matched[target_snapshot_column]
            - matched[target_price_date_column]
        ).dt.days

        valid_mask = (
            matched[target_price_date_column].notna()
            & matched[target_close_column].notna()
            & (
                matched[target_close_column]
                > 0
            )
            & (
                matched[target_price_date_column]
                > matched["signal_price_date"]
            )
            & (
                stale_days
                <= MAXIMUM_TARGET_PRICE_STALENESS_DAYS
            )
        )

        valid_rows = matched.loc[
            valid_mask
        ]

        if valid_rows.empty:
            continue

        destination_indices = valid_rows[
            "_panel_index"
        ].astype(int)

        panel.loc[
            destination_indices,
            target_price_date_column,
        ] = valid_rows[
            target_price_date_column
        ].to_numpy()

        panel.loc[
            destination_indices,
            target_close_column,
        ] = valid_rows[
            target_close_column
        ].to_numpy()

    panel[return_column] = (
        panel[target_close_column]
        / panel["close"]
        - 1.0
    )

    panel[return_column] = panel[
        return_column
    ].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return panel


def build_forward_return_panel(
    factor_panel: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    panel = add_target_snapshot_dates(
        factor_panel
    )

    for horizon_months in HORIZONS:
        print(
            f"Attaching forward "
            f"{horizon_months}-month prices...",
            flush=True,
        )

        panel = attach_one_horizon_prices(
            panel=panel,
            prices=prices,
            horizon_months=horizon_months,
        )

    return panel


def spearman_rank_correlation(
    scores: pd.Series,
    returns: pd.Series,
) -> float:
    valid_mask = (
        scores.notna()
        & returns.notna()
    )

    valid_scores = scores.loc[
        valid_mask
    ].astype(float)

    valid_returns = returns.loc[
        valid_mask
    ].astype(float)

    if len(valid_scores) < MINIMUM_IC_STOCKS:
        return np.nan

    if (
        valid_scores.nunique() < 3
        or valid_returns.nunique() < 3
    ):
        return np.nan

    score_ranks = valid_scores.rank(
        method="average"
    )

    return_ranks = valid_returns.rank(
        method="average"
    )

    return float(
        score_ranks.corr(return_ranks)
    )


def calculate_monthly_ic_and_quantiles(
    panel: pd.DataFrame,
    signal_metadata: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    monthly_ic_rows: list[dict] = []
    quantile_rows: list[dict] = []

    snapshot_groups = list(
        panel.groupby(
            "snapshot_date",
            sort=True,
        )
    )

    total_signals = len(signal_metadata)

    for signal_position, signal_info in (
        signal_metadata.iterrows()
    ):
        signal_column = signal_info[
            "signal_column"
        ]

        print(
            f"[{signal_position + 1}/"
            f"{total_signals}] Evaluating "
            f"{signal_column}",
            flush=True,
        )

        for horizon_months, return_column in (
            HORIZONS.items()
        ):
            target_snapshot_column = (
                f"target_snapshot_date_"
                f"{horizon_months}m"
            )

            for (
                snapshot_date,
                snapshot,
            ) in snapshot_groups:
                valid = snapshot[
                    [
                        signal_column,
                        return_column,
                        target_snapshot_column,
                    ]
                ].dropna(
                    subset=[
                        signal_column,
                        return_column,
                    ]
                )

                number_of_stocks = len(valid)

                if number_of_stocks < (
                    MINIMUM_IC_STOCKS
                ):
                    continue

                rank_ic = (
                    spearman_rank_correlation(
                        scores=valid[
                            signal_column
                        ],
                        returns=valid[
                            return_column
                        ],
                    )
                )

                monthly_ic_rows.append(
                    {
                        "snapshot_date": (
                            snapshot_date
                        ),
                        "target_snapshot_date": (
                            valid[
                                target_snapshot_column
                            ].iloc[0]
                        ),
                        "horizon_months": (
                            horizon_months
                        ),
                        "signal_column": (
                            signal_column
                        ),
                        "base_factor": signal_info[
                            "base_factor"
                        ],
                        "factor_name": signal_info[
                            "factor_name"
                        ],
                        "category": signal_info[
                            "category"
                        ],
                        "signal_variant": signal_info[
                            "signal_variant"
                        ],
                        "number_of_stocks": (
                            number_of_stocks
                        ),
                        "rank_ic": rank_ic,
                    }
                )

                if number_of_stocks < (
                    MINIMUM_QUANTILE_STOCKS
                ):
                    continue

                valid = valid.copy()

                ranked_scores = valid[
                    signal_column
                ].rank(
                    method="first"
                )

                try:
                    valid["quantile"] = pd.qcut(
                        ranked_scores,
                        q=5,
                        labels=[
                            1,
                            2,
                            3,
                            4,
                            5,
                        ],
                    ).astype(int)

                except ValueError:
                    continue

                quantile_returns = valid.groupby(
                    "quantile",
                    observed=True,
                )[return_column].agg(
                    [
                        "mean",
                        "count",
                    ]
                )

                for quantile, row in (
                    quantile_returns.iterrows()
                ):
                    quantile_rows.append(
                        {
                            "snapshot_date": (
                                snapshot_date
                            ),
                            "horizon_months": (
                                horizon_months
                            ),
                            "signal_column": (
                                signal_column
                            ),
                            "base_factor": (
                                signal_info[
                                    "base_factor"
                                ]
                            ),
                            "factor_name": (
                                signal_info[
                                    "factor_name"
                                ]
                            ),
                            "category": (
                                signal_info[
                                    "category"
                                ]
                            ),
                            "signal_variant": (
                                signal_info[
                                    "signal_variant"
                                ]
                            ),
                            "quantile": int(
                                quantile
                            ),
                            "mean_forward_return": (
                                float(row["mean"])
                            ),
                            "number_of_stocks": int(
                                row["count"]
                            ),
                        }
                    )

    return (
        pd.DataFrame(monthly_ic_rows),
        pd.DataFrame(quantile_rows),
    )


def summarize_ic(
    monthly_ic: pd.DataFrame,
) -> pd.DataFrame:
    group_columns = [
        "horizon_months",
        "signal_column",
        "base_factor",
        "factor_name",
        "category",
        "signal_variant",
    ]

    summary_rows: list[dict] = []

    for group_values, group in monthly_ic.groupby(
        group_columns,
        sort=False,
        dropna=False,
    ):
        ic_values = group[
            "rank_ic"
        ].dropna()

        number_of_periods = len(ic_values)

        mean_ic = ic_values.mean()
        ic_standard_deviation = (
            ic_values.std(ddof=1)
        )

        if (
            number_of_periods > 1
            and pd.notna(
                ic_standard_deviation
            )
            and ic_standard_deviation > 1e-12
        ):
            ic_ir = (
                mean_ic
                / ic_standard_deviation
            )

            t_statistic = (
                mean_ic
                / (
                    ic_standard_deviation
                    / math.sqrt(
                        number_of_periods
                    )
                )
            )
        else:
            ic_ir = np.nan
            t_statistic = np.nan

        positive_rate = (
            (ic_values > 0).mean() * 100
            if number_of_periods
            else np.nan
        )

        (
            horizon_months,
            signal_column,
            base_factor,
            factor_name,
            category,
            signal_variant,
        ) = group_values

        summary_rows.append(
            {
                "horizon_months": (
                    horizon_months
                ),
                "signal_column": signal_column,
                "base_factor": base_factor,
                "factor_name": factor_name,
                "category": category,
                "signal_variant": (
                    signal_variant
                ),
                "number_of_periods": (
                    number_of_periods
                ),
                "average_stocks_per_period": (
                    group[
                        "number_of_stocks"
                    ].mean()
                ),
                "mean_ic": mean_ic,
                "ic_standard_deviation": (
                    ic_standard_deviation
                ),
                "ic_ir": ic_ir,
                "positive_ic_rate_percent": (
                    positive_rate
                ),
                "t_statistic": t_statistic,
            }
        )

    return pd.DataFrame(
        summary_rows
    ).sort_values(
        [
            "horizon_months",
            "mean_ic",
        ],
        ascending=[
            True,
            False,
        ],
    ).reset_index(drop=True)


def summarize_quantiles(
    monthly_quantiles: pd.DataFrame,
) -> pd.DataFrame:
    group_columns = [
        "horizon_months",
        "signal_column",
        "base_factor",
        "factor_name",
        "category",
        "signal_variant",
    ]

    summary_rows: list[dict] = []

    for group_values, group in (
        monthly_quantiles.groupby(
            group_columns,
            sort=False,
            dropna=False,
        )
    ):
        average_quantile_returns = (
            group.groupby(
                "quantile"
            )["mean_forward_return"]
            .mean()
        )

        monthly_pivot = group.pivot_table(
            index="snapshot_date",
            columns="quantile",
            values="mean_forward_return",
            aggfunc="mean",
        )

        if (
            1 in monthly_pivot.columns
            and 5 in monthly_pivot.columns
        ):
            monthly_spread = (
                monthly_pivot[5]
                - monthly_pivot[1]
            ).dropna()
        else:
            monthly_spread = pd.Series(
                dtype=float
            )

        quantile_numbers = (
            average_quantile_returns.index
            .to_series()
            .astype(float)
        )

        quantile_values = (
            average_quantile_returns
            .astype(float)
        )

        if len(quantile_values) >= 3:
            monotonicity = (
                quantile_numbers.rank()
                .corr(
                    quantile_values.rank()
                )
            )
        else:
            monotonicity = np.nan

        (
            horizon_months,
            signal_column,
            base_factor,
            factor_name,
            category,
            signal_variant,
        ) = group_values

        summary_row = {
            "horizon_months": horizon_months,
            "signal_column": signal_column,
            "base_factor": base_factor,
            "factor_name": factor_name,
            "category": category,
            "signal_variant": signal_variant,
            "number_of_periods": int(
                group[
                    "snapshot_date"
                ].nunique()
            ),
            "quantile_monotonicity": (
                monotonicity
            ),
            "average_q5_minus_q1_return": (
                monthly_spread.mean()
                if not monthly_spread.empty
                else np.nan
            ),
            "q5_minus_q1_positive_rate_percent": (
                (monthly_spread > 0).mean()
                * 100
                if not monthly_spread.empty
                else np.nan
            ),
        }

        for quantile in range(1, 6):
            summary_row[
                f"average_q{quantile}_return"
            ] = average_quantile_returns.get(
                quantile,
                np.nan,
            )

        summary_rows.append(summary_row)

    return pd.DataFrame(
        summary_rows
    ).sort_values(
        [
            "horizon_months",
            "average_q5_minus_q1_return",
        ],
        ascending=[
            True,
            False,
        ],
    ).reset_index(drop=True)


def build_factor_decay_summary(
    ic_summary: pd.DataFrame,
) -> pd.DataFrame:
    identity_columns = [
        "signal_column",
        "base_factor",
        "factor_name",
        "category",
        "signal_variant",
    ]

    rows: list[dict] = []

    for identity_values, group in (
        ic_summary.groupby(
            identity_columns,
            sort=False,
            dropna=False,
        )
    ):
        (
            signal_column,
            base_factor,
            factor_name,
            category,
            signal_variant,
        ) = identity_values

        one_month = group.loc[
            group["horizon_months"] == 1
        ]

        three_month = group.loc[
            group["horizon_months"] == 3
        ]

        mean_ic_1m = (
            one_month["mean_ic"].iloc[0]
            if not one_month.empty
            else np.nan
        )

        mean_ic_3m = (
            three_month["mean_ic"].iloc[0]
            if not three_month.empty
            else np.nan
        )

        if (
            pd.notna(mean_ic_1m)
            and abs(mean_ic_1m) > 1e-12
            and pd.notna(mean_ic_3m)
        ):
            ic_retention_ratio = (
                mean_ic_3m / mean_ic_1m
            )
        else:
            ic_retention_ratio = np.nan

        rows.append(
            {
                "signal_column": signal_column,
                "base_factor": base_factor,
                "factor_name": factor_name,
                "category": category,
                "signal_variant": signal_variant,
                "mean_ic_1m": mean_ic_1m,
                "mean_ic_3m": mean_ic_3m,
                "ic_retention_ratio": (
                    ic_retention_ratio
                ),
            }
        )

    return pd.DataFrame(rows).sort_values(
        "mean_ic_1m",
        ascending=False,
    ).reset_index(drop=True)


def format_date_columns(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    output = frame.copy()

    date_columns = [
        column
        for column in output.columns
        if (
            column.endswith("_date")
            or column == "snapshot_date"
        )
    ]

    for column in date_columns:
        output[column] = pd.to_datetime(
            output[column],
            errors="coerce",
        ).dt.strftime("%Y-%m-%d")

    return output


def save_outputs(
    connection: sqlite3.Connection,
    panel: pd.DataFrame,
    monthly_ic: pd.DataFrame,
    ic_summary: pd.DataFrame,
    monthly_quantiles: pd.DataFrame,
    quantile_summary: pd.DataFrame,
    factor_decay: pd.DataFrame,
) -> None:
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    forward_columns = [
        "snapshot_date",
        "signal_price_date",
        "baostock_code",
        "symbol",
        "stock_name",
        "industry",
        "close",
        "target_snapshot_date_1m",
        "target_price_date_1m",
        "target_close_1m",
        "forward_return_1m",
        "target_snapshot_date_3m",
        "target_price_date_3m",
        "target_close_3m",
        "forward_return_3m",
    ]

    forward_columns = [
        column
        for column in forward_columns
        if column in panel.columns
    ]

    forward_returns = format_date_columns(
        panel[forward_columns]
    )

    monthly_ic_output = format_date_columns(
        monthly_ic
    )

    monthly_quantile_output = (
        format_date_columns(
            monthly_quantiles
        )
    )

    output_items = [
        (
            FORWARD_RETURN_TABLE,
            forward_returns,
            FORWARD_RETURN_FILE,
        ),
        (
            MONTHLY_IC_TABLE,
            monthly_ic_output,
            MONTHLY_IC_FILE,
        ),
        (
            IC_SUMMARY_TABLE,
            ic_summary,
            IC_SUMMARY_FILE,
        ),
        (
            MONTHLY_QUANTILE_TABLE,
            monthly_quantile_output,
            MONTHLY_QUANTILE_FILE,
        ),
        (
            QUANTILE_SUMMARY_TABLE,
            quantile_summary,
            QUANTILE_SUMMARY_FILE,
        ),
        (
            FACTOR_DECAY_TABLE,
            factor_decay,
            FACTOR_DECAY_FILE,
        ),
    ]

    for (
        table_name,
        output_frame,
        output_file,
    ) in output_items:
        output_frame.to_sql(
            table_name,
            connection,
            if_exists="replace",
            index=False,
            chunksize=2_000,
        )

        output_frame.to_csv(
            output_file,
            index=False,
            encoding="utf-8-sig",
            float_format="%.10f",
        )

    connection.execute(
        f"""
        CREATE INDEX IF NOT EXISTS
        idx_{MONTHLY_IC_TABLE}_factor_horizon
        ON {MONTHLY_IC_TABLE} (
            signal_column,
            horizon_months
        );
        """
    )

    connection.commit()


def print_forward_return_coverage(
    panel: pd.DataFrame,
) -> None:
    print()
    print("=" * 120)
    print("FORWARD-RETURN COVERAGE")
    print("=" * 120)

    rows = []

    for horizon_months, return_column in (
        HORIZONS.items()
    ):
        valid_count = int(
            panel[return_column].notna().sum()
        )

        total_count = len(panel)

        coverage_percent = (
            valid_count / total_count * 100
            if total_count
            else 0.0
        )

        valid_snapshots = panel.loc[
            panel[return_column].notna(),
            "snapshot_date",
        ]

        rows.append(
            {
                "horizon": (
                    f"{horizon_months} month"
                ),
                "valid_observations": (
                    valid_count
                ),
                "total_observations": (
                    total_count
                ),
                "coverage_percent": (
                    coverage_percent
                ),
                "last_evaluable_snapshot": (
                    valid_snapshots.max()
                    if not valid_snapshots.empty
                    else pd.NaT
                ),
            }
        )

    coverage = pd.DataFrame(rows)

    print(
        coverage.to_string(
            index=False,
            formatters={
                "coverage_percent": (
                    lambda value: f"{value:.2f}%"
                ),
                "last_evaluable_snapshot": (
                    lambda value: (
                        ""
                        if pd.isna(value)
                        else value.date().isoformat()
                    )
                ),
            },
        )
    )


def print_ic_summary(
    ic_summary: pd.DataFrame,
) -> None:
    selected = ic_summary.loc[
        (
            ic_summary["horizon_months"]
            == 1
        )
        & (
            ic_summary["signal_variant"]
            != "standardized"
        )
    ].copy()

    selected = selected.sort_values(
        "mean_ic",
        ascending=False,
    ).head(20)

    display_columns = [
        "factor_name",
        "signal_variant",
        "number_of_periods",
        "mean_ic",
        "ic_ir",
        "positive_ic_rate_percent",
        "t_statistic",
    ]

    print()
    print("=" * 120)
    print(
        "TOP 20 ONE-MONTH NEUTRALIZED "
        "AND COMPOSITE IC RESULTS"
    )
    print("=" * 120)

    print(
        selected[display_columns].to_string(
            index=False,
            formatters={
                "mean_ic": (
                    lambda value: f"{value:.4f}"
                ),
                "ic_ir": (
                    lambda value: f"{value:.4f}"
                ),
                "positive_ic_rate_percent": (
                    lambda value: f"{value:.2f}%"
                ),
                "t_statistic": (
                    lambda value: f"{value:.4f}"
                ),
            },
        )
    )


def print_quantile_summary(
    quantile_summary: pd.DataFrame,
) -> None:
    selected = quantile_summary.loc[
        (
            quantile_summary[
                "horizon_months"
            ]
            == 1
        )
        & (
            quantile_summary[
                "signal_variant"
            ]
            != "standardized"
        )
    ].copy()

    selected = selected.sort_values(
        "average_q5_minus_q1_return",
        ascending=False,
    ).head(15)

    display_columns = [
        "factor_name",
        "signal_variant",
        "average_q1_return",
        "average_q5_return",
        "average_q5_minus_q1_return",
        "q5_minus_q1_positive_rate_percent",
        "quantile_monotonicity",
    ]

    percent_columns = [
        "average_q1_return",
        "average_q5_return",
        "average_q5_minus_q1_return",
    ]

    formatters = {
        column: (
            lambda value: f"{value * 100:.2f}%"
        )
        for column in percent_columns
    }

    formatters[
        "q5_minus_q1_positive_rate_percent"
    ] = lambda value: f"{value:.2f}%"

    formatters[
        "quantile_monotonicity"
    ] = lambda value: f"{value:.4f}"

    print()
    print("=" * 120)
    print(
        "TOP 15 ONE-MONTH QUANTILE SPREADS"
    )
    print("=" * 120)

    print(
        selected[display_columns].to_string(
            index=False,
            formatters=formatters,
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
    signal_metadata = build_signal_metadata()

    print("=" * 120)
    print("V2 FACTOR EVALUATION")
    print("=" * 120)
    print(f"Database: {database_path}")
    print(
        f"Signals to evaluate: "
        f"{len(signal_metadata)}"
    )
    print()

    with sqlite3.connect(
        database_path
    ) as connection:
        for table_name in [
            FACTOR_TABLE,
            PRICE_TABLE,
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
            "Loading processed factors...",
            flush=True,
        )
        factor_panel = load_processed_factors(
            connection,
            signal_metadata,
        )

        print(
            f"Factor observations: "
            f"{len(factor_panel):,}"
        )
        print(
            f"Factor snapshots: "
            f"{factor_panel['snapshot_date'].nunique():,}"
        )

        print(
            "Loading daily prices...",
            flush=True,
        )
        prices = load_daily_prices(connection)

        print(
            f"Daily price observations: "
            f"{len(prices):,}"
        )

        panel = build_forward_return_panel(
            factor_panel=factor_panel,
            prices=prices,
        )

        print_forward_return_coverage(panel)

        print()
        print(
            "Calculating IC and quantile returns...",
            flush=True,
        )

        (
            monthly_ic,
            monthly_quantiles,
        ) = calculate_monthly_ic_and_quantiles(
            panel=panel,
            signal_metadata=signal_metadata,
        )

        ic_summary = summarize_ic(
            monthly_ic
        )

        quantile_summary = (
            summarize_quantiles(
                monthly_quantiles
            )
        )

        factor_decay = (
            build_factor_decay_summary(
                ic_summary
            )
        )

        print(
            "Saving factor-evaluation results...",
            flush=True,
        )

        save_outputs(
            connection=connection,
            panel=panel,
            monthly_ic=monthly_ic,
            ic_summary=ic_summary,
            monthly_quantiles=(
                monthly_quantiles
            ),
            quantile_summary=(
                quantile_summary
            ),
            factor_decay=factor_decay,
        )

    print_ic_summary(ic_summary)
    print_quantile_summary(
        quantile_summary
    )

    print()
    print("=" * 120)
    print("OUTPUT FILES")
    print("=" * 120)
    print(FORWARD_RETURN_FILE)
    print(MONTHLY_IC_FILE)
    print(IC_SUMMARY_FILE)
    print(MONTHLY_QUANTILE_FILE)
    print(QUANTILE_SUMMARY_FILE)
    print(FACTOR_DECAY_FILE)
    print()
    print("V2 factor evaluation completed.")


if __name__ == "__main__":
    main()