"""
V2 cross-sectional factor processing.

Processing sequence for every monthly snapshot:

1. Replace infinite values with missing values
2. Fill missing values using industry median
3. Fill remaining missing values using cross-sectional median
4. MAD winsorization
5. Apply factor direction
6. Cross-sectional Z-score standardization
7. Industry and market-cap neutralization
8. Re-standardize neutralized residuals
9. Build category scores
10. Build an equal-category-weight composite score

All operations are performed within the same snapshot to avoid look-ahead bias.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from src.factors_v2 import (
    FACTOR_METADATA,
    resolve_database_path,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "reports" / "results"

INPUT_TABLE = "factor_values_v2"
OUTPUT_TABLE = "processed_factor_values_v2"

PROCESSED_FACTOR_FILE = (
    RESULTS_DIR / "processed_factors_v2.csv"
)
PROCESSING_SUMMARY_FILE = (
    RESULTS_DIR / "factor_processing_summary_v2.csv"
)
NEUTRALITY_DIAGNOSTICS_FILE = (
    RESULTS_DIR / "factor_neutrality_diagnostics_v2.csv"
)


ALPHA_METADATA = [
    item
    for item in FACTOR_METADATA
    if item["role"] == "alpha"
]

ALPHA_FACTORS = [
    item["factor_column"]
    for item in ALPHA_METADATA
]

FACTOR_DIRECTIONS = {
    item["factor_column"]: int(item["direction"])
    for item in ALPHA_METADATA
}

SIZE_COLUMN = "log_float_market_cap"


CATEGORY_DEFINITIONS = {
    "momentum_category_score": [
        "momentum_20",
        "momentum_60",
        "momentum_120",
        "momentum_12_1",
    ],
    "reversal_category_score": [
        "reversal_5_score",
    ],
    "risk_category_score": [
        "volatility_20",
        "downside_volatility_60",
        "drawdown_60",
    ],
    "trend_category_score": [
        "trend_60",
    ],
    "liquidity_category_score": [
        "liquidity_amount_20",
        "turnover_20",
        "turnover_stability_20",
        "amihud_liquidity_score_20",
    ],
    "value_category_score": [
        "value_pe_score",
        "value_pb_score",
        "value_ps_score",
        "value_pcf_score",
    ],
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


def load_raw_factors(
    connection: sqlite3.Connection,
) -> pd.DataFrame:
    factors = pd.read_sql_query(
        f'SELECT * FROM "{INPUT_TABLE}"',
        connection,
    )

    required_columns = {
        "snapshot_date",
        "signal_price_date",
        "baostock_code",
        "industry",
        SIZE_COLUMN,
        *ALPHA_FACTORS,
    }

    missing_columns = required_columns.difference(
        factors.columns
    )

    if missing_columns:
        raise ValueError(
            f"{INPUT_TABLE} is missing required columns: "
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
        SIZE_COLUMN,
        *ALPHA_FACTORS,
    ]

    for column in numeric_columns:
        if column in factors.columns:
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

    factors["industry"] = (
        factors["industry"]
        .fillna("UNKNOWN")
        .astype(str)
        .str.strip()
        .replace("", "UNKNOWN")
    )

    factors = factors.dropna(
        subset=[
            "snapshot_date",
            "signal_price_date",
            "baostock_code",
        ]
    )

    duplicate_count = int(
        factors.duplicated(
            ["snapshot_date", "baostock_code"]
        ).sum()
    )

    if duplicate_count:
        raise ValueError(
            f"{INPUT_TABLE} contains "
            f"{duplicate_count:,} duplicate snapshot-stock rows."
        )

    return factors.sort_values(
        ["snapshot_date", "baostock_code"]
    ).reset_index(drop=True)


def fill_missing_values(
    values: pd.Series,
    industries: pd.Series,
) -> tuple[pd.Series, dict[str, int]]:
    values = pd.to_numeric(
        values,
        errors="coerce",
    ).replace(
        [np.inf, -np.inf],
        np.nan,
    )

    original_missing = values.isna()

    industry_medians = values.groupby(
        industries,
        dropna=False,
    ).transform("median")

    after_industry = values.fillna(
        industry_medians
    )

    industry_filled = (
        original_missing
        & after_industry.notna()
    )

    cross_section_median = after_industry.median()

    if pd.notna(cross_section_median):
        after_cross_section = (
            after_industry.fillna(
                cross_section_median
            )
        )
    else:
        after_cross_section = after_industry.copy()

    cross_section_filled = (
        after_industry.isna()
        & after_cross_section.notna()
    )

    remaining_missing = after_cross_section.isna()

    final_values = after_cross_section.fillna(0.0)

    statistics = {
        "raw_missing_observations": int(
            original_missing.sum()
        ),
        "industry_imputed_observations": int(
            industry_filled.sum()
        ),
        "cross_section_imputed_observations": int(
            cross_section_filled.sum()
        ),
        "zero_fallback_observations": int(
            remaining_missing.sum()
        ),
    }

    return final_values.astype(float), statistics


def mad_winsorize(
    values: pd.Series,
    mad_multiplier: float = 5.0,
) -> tuple[pd.Series, int]:
    values = values.astype(float)

    median = values.median()
    mad = (values - median).abs().median()

    if pd.notna(mad) and mad > 1e-12:
        robust_scale = 1.4826 * mad

        lower_bound = (
            median
            - mad_multiplier * robust_scale
        )
        upper_bound = (
            median
            + mad_multiplier * robust_scale
        )
    else:
        lower_bound = values.quantile(0.01)
        upper_bound = values.quantile(0.99)

    if (
        pd.isna(lower_bound)
        or pd.isna(upper_bound)
        or lower_bound > upper_bound
    ):
        return values.copy(), 0

    clipped_mask = (
        (values < lower_bound)
        | (values > upper_bound)
    )

    winsorized = values.clip(
        lower=lower_bound,
        upper=upper_bound,
    )

    return winsorized, int(clipped_mask.sum())


def zscore(
    values: pd.Series,
) -> pd.Series:
    values = values.astype(float)

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


def safe_correlation(
    first: pd.Series,
    second: pd.Series,
) -> float:
    valid_mask = (
        first.notna()
        & second.notna()
    )

    first_valid = first.loc[valid_mask].astype(float)
    second_valid = second.loc[valid_mask].astype(float)

    if len(first_valid) < 3:
        return 0.0

    if (
        first_valid.std(ddof=0) < 1e-12
        or second_valid.std(ddof=0) < 1e-12
    ):
        return 0.0

    correlation = first_valid.corr(
        second_valid
    )

    if pd.isna(correlation):
        return 0.0

    return float(correlation)


def neutralize_factor(
    values: pd.Series,
    industries: pd.Series,
    size_zscore: pd.Series,
) -> pd.Series:
    """
    Regress factor values on:

    - Intercept
    - Log market-cap Z-score
    - Industry dummy variables

    The regression residual is the neutralized factor.
    """

    industry_text = (
        industries
        .fillna("UNKNOWN")
        .astype(str)
        .replace("", "UNKNOWN")
    )

    industry_dummies = pd.get_dummies(
        industry_text,
        prefix="industry",
        dtype=float,
    )

    if industry_dummies.shape[1] > 1:
        industry_dummies = (
            industry_dummies.iloc[:, 1:]
        )
    else:
        industry_dummies = pd.DataFrame(
            index=values.index
        )

    design_parts = [
        pd.Series(
            1.0,
            index=values.index,
            name="intercept",
        ),
        size_zscore.rename("size_zscore"),
    ]

    design_matrix = pd.concat(
        [
            *design_parts,
            industry_dummies,
        ],
        axis=1,
    ).astype(float)

    valid_mask = (
        values.notna()
        & design_matrix.notna().all(axis=1)
    )

    residuals = pd.Series(
        np.nan,
        index=values.index,
        dtype=float,
    )

    if valid_mask.sum() <= design_matrix.shape[1]:
        residuals.loc[valid_mask] = (
            values.loc[valid_mask]
        )
        return residuals.fillna(0.0)

    y = values.loc[valid_mask].to_numpy(
        dtype=float
    )

    x = design_matrix.loc[
        valid_mask
    ].to_numpy(dtype=float)

    coefficients, _, _, _ = np.linalg.lstsq(
        x,
        y,
        rcond=None,
    )

    fitted_values = x @ coefficients

    residuals.loc[valid_mask] = (
        y - fitted_values
    )

    residuals = residuals.fillna(0.0)

    return residuals


def mean_absolute_industry_score(
    values: pd.Series,
    industries: pd.Series,
) -> float:
    industry_means = values.groupby(
        industries,
        dropna=False,
    ).mean()

    if industry_means.empty:
        return 0.0

    return float(
        industry_means.abs().mean()
    )


def create_empty_statistics() -> dict[str, int]:
    return {
        "total_observations": 0,
        "raw_missing_observations": 0,
        "industry_imputed_observations": 0,
        "cross_section_imputed_observations": 0,
        "zero_fallback_observations": 0,
        "winsorized_observations": 0,
    }


def update_statistics(
    accumulator: dict[str, int],
    new_statistics: dict[str, int],
) -> None:
    for key, value in new_statistics.items():
        accumulator[key] += int(value)


def process_one_snapshot(
    snapshot: pd.DataFrame,
    processing_statistics: dict[str, dict[str, int]],
) -> tuple[pd.DataFrame, list[dict]]:
    snapshot = snapshot.copy()

    snapshot_date = snapshot[
        "snapshot_date"
    ].iloc[0]

    industries = snapshot["industry"]

    size_filled, size_missing_statistics = (
        fill_missing_values(
            values=snapshot[SIZE_COLUMN],
            industries=industries,
        )
    )

    size_winsorized, size_winsorized_count = (
        mad_winsorize(size_filled)
    )

    snapshot["size_zscore"] = zscore(
        size_winsorized
    )

    size_statistics = {
        "total_observations": len(snapshot),
        **size_missing_statistics,
        "winsorized_observations": (
            size_winsorized_count
        ),
    }

    update_statistics(
        processing_statistics[SIZE_COLUMN],
        size_statistics,
    )

    diagnostics: list[dict] = []

    for factor_column in ALPHA_FACTORS:
        filled_values, missing_statistics = (
            fill_missing_values(
                values=snapshot[factor_column],
                industries=industries,
            )
        )

        winsorized_values, winsorized_count = (
            mad_winsorize(filled_values)
        )

        direction = FACTOR_DIRECTIONS[
            factor_column
        ]

        directed_values = (
            winsorized_values * direction
        )

        standardized_values = zscore(
            directed_values
        )

        neutralized_residuals = (
            neutralize_factor(
                values=standardized_values,
                industries=industries,
                size_zscore=snapshot[
                    "size_zscore"
                ],
            )
        )

        neutralized_values = zscore(
            neutralized_residuals
        )

        zscore_column = (
            f"{factor_column}_zscore"
        )

        neutral_column = (
            f"{factor_column}_neutral"
        )

        snapshot[zscore_column] = (
            standardized_values
        )

        snapshot[neutral_column] = (
            neutralized_values
        )

        factor_statistics = {
            "total_observations": len(snapshot),
            **missing_statistics,
            "winsorized_observations": (
                winsorized_count
            ),
        }

        update_statistics(
            processing_statistics[factor_column],
            factor_statistics,
        )

        diagnostics.append(
            {
                "snapshot_date": snapshot_date,
                "factor_column": factor_column,
                "number_of_stocks": len(snapshot),
                "size_correlation_before": (
                    safe_correlation(
                        standardized_values,
                        snapshot["size_zscore"],
                    )
                ),
                "size_correlation_after": (
                    safe_correlation(
                        neutralized_values,
                        snapshot["size_zscore"],
                    )
                ),
                "mean_absolute_industry_score_before": (
                    mean_absolute_industry_score(
                        standardized_values,
                        industries,
                    )
                ),
                "mean_absolute_industry_score_after": (
                    mean_absolute_industry_score(
                        neutralized_values,
                        industries,
                    )
                ),
            }
        )

    for (
        category_score_column,
        category_factors,
    ) in CATEGORY_DEFINITIONS.items():
        neutral_columns = [
            f"{factor}_neutral"
            for factor in category_factors
        ]

        raw_category_score = snapshot[
            neutral_columns
        ].mean(axis=1)

        snapshot[category_score_column] = zscore(
            raw_category_score
        )

    category_score_columns = list(
        CATEGORY_DEFINITIONS.keys()
    )

    raw_composite_score = snapshot[
        category_score_columns
    ].mean(axis=1)

    snapshot[
        "equal_weight_composite_score"
    ] = zscore(raw_composite_score)

    return snapshot, diagnostics


def process_all_snapshots(
    raw_factors: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, dict[str, int]],
]:
    processing_statistics = {
        column: create_empty_statistics()
        for column in [
            SIZE_COLUMN,
            *ALPHA_FACTORS,
        ]
    }

    processed_snapshots: list[pd.DataFrame] = []
    all_diagnostics: list[dict] = []

    grouped_snapshots = list(
        raw_factors.groupby(
            "snapshot_date",
            sort=True,
        )
    )

    total_snapshots = len(grouped_snapshots)

    for position, (
        snapshot_date,
        snapshot,
    ) in enumerate(
        grouped_snapshots,
        start=1,
    ):
        processed_snapshot, diagnostics = (
            process_one_snapshot(
                snapshot=snapshot,
                processing_statistics=(
                    processing_statistics
                ),
            )
        )

        processed_snapshots.append(
            processed_snapshot
        )
        all_diagnostics.extend(diagnostics)

        if (
            position == 1
            or position % 10 == 0
            or position == total_snapshots
        ):
            print(
                f"[{position}/{total_snapshots}] "
                f"Processed snapshot "
                f"{snapshot_date.date()}",
                flush=True,
            )

    processed = pd.concat(
        processed_snapshots,
        ignore_index=True,
    )

    diagnostics_frame = pd.DataFrame(
        all_diagnostics
    )

    return (
        processed,
        diagnostics_frame,
        processing_statistics,
    )


def build_processing_summary(
    processing_statistics: dict[
        str,
        dict[str, int],
    ],
    diagnostics: pd.DataFrame,
) -> pd.DataFrame:
    metadata = pd.DataFrame(
        FACTOR_METADATA
    )

    rows: list[dict] = []

    for factor_column, statistics in (
        processing_statistics.items()
    ):
        total = statistics[
            "total_observations"
        ]

        if total > 0:
            raw_missing_percent = (
                statistics[
                    "raw_missing_observations"
                ]
                / total
                * 100
            )

            winsorized_percent = (
                statistics[
                    "winsorized_observations"
                ]
                / total
                * 100
            )
        else:
            raw_missing_percent = 0.0
            winsorized_percent = 0.0

        factor_diagnostics = diagnostics.loc[
            diagnostics["factor_column"]
            == factor_column
        ]

        if factor_diagnostics.empty:
            mean_absolute_size_before = np.nan
            mean_absolute_size_after = np.nan
            mean_industry_after = np.nan
        else:
            mean_absolute_size_before = (
                factor_diagnostics[
                    "size_correlation_before"
                ].abs().mean()
            )

            mean_absolute_size_after = (
                factor_diagnostics[
                    "size_correlation_after"
                ].abs().mean()
            )

            mean_industry_after = (
                factor_diagnostics[
                    "mean_absolute_industry_score_after"
                ].mean()
            )

        rows.append(
            {
                "factor_column": factor_column,
                **statistics,
                "raw_missing_percent": (
                    raw_missing_percent
                ),
                "winsorized_percent": (
                    winsorized_percent
                ),
                "mean_absolute_size_correlation_before": (
                    mean_absolute_size_before
                ),
                "mean_absolute_size_correlation_after": (
                    mean_absolute_size_after
                ),
                "mean_absolute_industry_score_after": (
                    mean_industry_after
                ),
            }
        )

    summary = pd.DataFrame(rows)

    summary = metadata.merge(
        summary,
        on="factor_column",
        how="right",
        validate="one_to_one",
    )

    return summary


def select_output_columns(
    processed: pd.DataFrame,
) -> pd.DataFrame:
    zscore_columns = [
        f"{factor}_zscore"
        for factor in ALPHA_FACTORS
    ]

    neutral_columns = [
        f"{factor}_neutral"
        for factor in ALPHA_FACTORS
    ]

    category_columns = list(
        CATEGORY_DEFINITIONS.keys()
    )

    output_columns = [
        *IDENTIFIER_COLUMNS,
        SIZE_COLUMN,
        "size_zscore",
        *zscore_columns,
        *neutral_columns,
        *category_columns,
        "equal_weight_composite_score",
    ]

    existing_columns = [
        column
        for column in output_columns
        if column in processed.columns
    ]

    return processed[
        existing_columns
    ].sort_values(
        ["snapshot_date", "baostock_code"]
    ).reset_index(drop=True)


def save_outputs(
    connection: sqlite3.Connection,
    processed: pd.DataFrame,
    summary: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> None:
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    database_output = processed.copy()
    diagnostics_output = diagnostics.copy()

    date_columns = [
        "snapshot_date",
        "signal_price_date",
    ]

    for column in date_columns:
        if column in database_output.columns:
            database_output[column] = (
                pd.to_datetime(
                    database_output[column]
                )
                .dt.strftime("%Y-%m-%d")
            )

    if "snapshot_date" in diagnostics_output.columns:
        diagnostics_output["snapshot_date"] = (
            pd.to_datetime(
                diagnostics_output[
                    "snapshot_date"
                ]
            )
            .dt.strftime("%Y-%m-%d")
        )

    database_output.to_sql(
        OUTPUT_TABLE,
        connection,
        if_exists="replace",
        index=False,
        chunksize=2_000,
    )

    connection.execute(
        f"""
        CREATE UNIQUE INDEX IF NOT EXISTS
        idx_{OUTPUT_TABLE}_snapshot_code
        ON {OUTPUT_TABLE} (
            snapshot_date,
            baostock_code
        );
        """
    )

    connection.execute(
        f"""
        CREATE INDEX IF NOT EXISTS
        idx_{OUTPUT_TABLE}_code
        ON {OUTPUT_TABLE} (
            baostock_code
        );
        """
    )

    connection.commit()

    database_output.to_csv(
        PROCESSED_FACTOR_FILE,
        index=False,
        encoding="utf-8-sig",
        float_format="%.10f",
    )

    summary.to_csv(
        PROCESSING_SUMMARY_FILE,
        index=False,
        encoding="utf-8-sig",
        float_format="%.8f",
    )

    diagnostics_output.to_csv(
        NEUTRALITY_DIAGNOSTICS_FILE,
        index=False,
        encoding="utf-8-sig",
        float_format="%.8f",
    )


def print_latest_ranking(
    processed: pd.DataFrame,
) -> None:
    latest_snapshot = processed[
        "snapshot_date"
    ].max()

    latest = processed.loc[
        processed["snapshot_date"]
        == latest_snapshot
    ].copy()

    latest = latest.sort_values(
        "equal_weight_composite_score",
        ascending=False,
    ).head(15)

    display_columns = [
        "baostock_code",
        "symbol",
        "stock_name",
        "industry",
        "equal_weight_composite_score",
    ]

    display_columns = [
        column
        for column in display_columns
        if column in latest.columns
    ]

    print()
    print("=" * 120)
    print(
        f"LATEST EQUAL-WEIGHT FACTOR RANKING: "
        f"{latest_snapshot.date()}"
    )
    print("=" * 120)

    print(
        latest[display_columns].to_string(
            index=False,
            formatters={
                "equal_weight_composite_score": (
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
        200,
    )

    database_path = resolve_database_path()

    print("=" * 120)
    print("V2 FACTOR PROCESSING AND NEUTRALIZATION")
    print("=" * 120)
    print(f"Database: {database_path}")
    print()

    with sqlite3.connect(
        database_path
    ) as connection:
        if not table_exists(
            connection,
            INPUT_TABLE,
        ):
            raise RuntimeError(
                f"Required table does not exist: "
                f"{INPUT_TABLE}"
            )

        print(
            "Loading raw monthly factors...",
            flush=True,
        )

        raw_factors = load_raw_factors(
            connection
        )

        print(
            f"Raw observations: "
            f"{len(raw_factors):,}"
        )
        print(
            f"Snapshots: "
            f"{raw_factors['snapshot_date'].nunique():,}"
        )
        print(
            f"Alpha factors: "
            f"{len(ALPHA_FACTORS):,}"
        )
        print()

        print(
            "Processing monthly cross-sections...",
            flush=True,
        )

        (
            processed,
            diagnostics,
            processing_statistics,
        ) = process_all_snapshots(
            raw_factors
        )

        processed = select_output_columns(
            processed
        )

        summary = build_processing_summary(
            processing_statistics=(
                processing_statistics
            ),
            diagnostics=diagnostics,
        )

        print(
            "Saving processed factor library...",
            flush=True,
        )

        save_outputs(
            connection=connection,
            processed=processed,
            summary=summary,
            diagnostics=diagnostics,
        )

    print()
    print("=" * 120)
    print("V2 FACTOR-PROCESSING SUMMARY")
    print("=" * 120)

    print(
        f"Processed observations: "
        f"{len(processed):,}"
    )
    print(
        f"Processed snapshots: "
        f"{processed['snapshot_date'].nunique():,}"
    )
    print(
        f"Neutralized alpha factors: "
        f"{len(ALPHA_FACTORS):,}"
    )
    print(
        f"Category scores: "
        f"{len(CATEGORY_DEFINITIONS):,}"
    )

    print()
    print("=" * 120)
    print("PROCESSING AND NEUTRALITY CHECK")
    print("=" * 120)

    display_columns = [
        "factor_column",
        "raw_missing_percent",
        "winsorized_percent",
        "mean_absolute_size_correlation_before",
        "mean_absolute_size_correlation_after",
        "mean_absolute_industry_score_after",
    ]

    print(
        summary[display_columns].to_string(
            index=False,
            formatters={
                "raw_missing_percent": (
                    lambda value: f"{value:.2f}%"
                ),
                "winsorized_percent": (
                    lambda value: f"{value:.2f}%"
                ),
                "mean_absolute_size_correlation_before": (
                    lambda value: (
                        ""
                        if pd.isna(value)
                        else f"{value:.4f}"
                    )
                ),
                "mean_absolute_size_correlation_after": (
                    lambda value: (
                        ""
                        if pd.isna(value)
                        else f"{value:.6f}"
                    )
                ),
                "mean_absolute_industry_score_after": (
                    lambda value: (
                        ""
                        if pd.isna(value)
                        else f"{value:.6f}"
                    )
                ),
            },
        )
    )

    print_latest_ranking(processed)

    print()
    print(f"Database table: {OUTPUT_TABLE}")
    print(
        f"Processed-factor CSV: "
        f"{PROCESSED_FACTOR_FILE}"
    )
    print(
        f"Processing summary: "
        f"{PROCESSING_SUMMARY_FILE}"
    )
    print(
        f"Neutrality diagnostics: "
        f"{NEUTRALITY_DIAGNOSTICS_FILE}"
    )
    print()
    print(
        "V2 factor processing completed."
    )


if __name__ == "__main__":
    main()