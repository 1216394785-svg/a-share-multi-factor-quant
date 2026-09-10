"""
V2 raw factor library.

This module calculates raw monthly factor values using:

1. daily_prices_v2
2. eligible_universe_v2
3. stock_master_v2

No cross-sectional standardization or neutralization is performed here.
Those operations will be handled by factor_processing_v2.py.
"""

from __future__ import annotations

import math
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "reports" / "results"

DAILY_TABLE = "daily_prices_v2"
ELIGIBLE_TABLE = "eligible_universe_v2"
MASTER_TABLE = "stock_master_v2"
OUTPUT_TABLE = "factor_values_v2"

RAW_FACTOR_FILE = RESULTS_DIR / "raw_factors_v2.csv"
COVERAGE_FILE = RESULTS_DIR / "factor_coverage_v2.csv"
METADATA_FILE = RESULTS_DIR / "factor_metadata_v2.csv"


FACTOR_METADATA = [
    {
        "factor_column": "momentum_20",
        "factor_name": "20-Day Momentum",
        "category": "Momentum",
        "direction": 1,
        "role": "alpha",
        "formula": "close / close.shift(20) - 1",
    },
    {
        "factor_column": "momentum_60",
        "factor_name": "60-Day Momentum",
        "category": "Momentum",
        "direction": 1,
        "role": "alpha",
        "formula": "close / close.shift(60) - 1",
    },
    {
        "factor_column": "momentum_120",
        "factor_name": "120-Day Momentum",
        "category": "Momentum",
        "direction": 1,
        "role": "alpha",
        "formula": "close / close.shift(120) - 1",
    },
    {
        "factor_column": "momentum_12_1",
        "factor_name": "12-1 Momentum",
        "category": "Momentum",
        "direction": 1,
        "role": "alpha",
        "formula": "close.shift(20) / close.shift(252) - 1",
    },
    {
        "factor_column": "reversal_5_score",
        "factor_name": "5-Day Reversal",
        "category": "Reversal",
        "direction": 1,
        "role": "alpha",
        "formula": "-(close / close.shift(5) - 1)",
    },
    {
        "factor_column": "volatility_20",
        "factor_name": "20-Day Volatility",
        "category": "Risk",
        "direction": -1,
        "role": "alpha",
        "formula": "std(daily return, 20) * sqrt(252)",
    },
    {
        "factor_column": "downside_volatility_60",
        "factor_name": "60-Day Downside Volatility",
        "category": "Risk",
        "direction": -1,
        "role": "alpha",
        "formula": "std(min(daily return, 0), 60) * sqrt(252)",
    },
    {
        "factor_column": "drawdown_60",
        "factor_name": "60-Day Drawdown",
        "category": "Risk",
        "direction": 1,
        "role": "alpha",
        "formula": "close / rolling_max(close, 60) - 1",
    },
    {
        "factor_column": "trend_60",
        "factor_name": "60-Day MA Trend",
        "category": "Trend",
        "direction": 1,
        "role": "alpha",
        "formula": "close / rolling_mean(close, 60) - 1",
    },
    {
        "factor_column": "liquidity_amount_20",
        "factor_name": "20-Day Amount Liquidity",
        "category": "Liquidity",
        "direction": 1,
        "role": "alpha",
        "formula": "log(1 + rolling_mean(amount, 20) / 1,000,000)",
    },
    {
        "factor_column": "turnover_20",
        "factor_name": "20-Day Turnover",
        "category": "Liquidity",
        "direction": 1,
        "role": "alpha",
        "formula": "rolling_mean(turnover_rate, 20)",
    },
    {
        "factor_column": "turnover_stability_20",
        "factor_name": "Turnover Stability",
        "category": "Liquidity",
        "direction": 1,
        "role": "alpha",
        "formula": "-rolling_std(turnover_rate, 20)",
    },
    {
        "factor_column": "amihud_liquidity_score_20",
        "factor_name": "Amihud Liquidity",
        "category": "Liquidity",
        "direction": 1,
        "role": "alpha",
        "formula": "-log(1 + rolling_mean(abs(return)/(amount/1e8), 20))",
    },
    {
        "factor_column": "value_pe_score",
        "factor_name": "Earnings Yield Style",
        "category": "Value",
        "direction": 1,
        "role": "alpha",
        "formula": "-log(positive PE TTM)",
    },
    {
        "factor_column": "value_pb_score",
        "factor_name": "Book-to-Price Style",
        "category": "Value",
        "direction": 1,
        "role": "alpha",
        "formula": "-log(positive PB MRQ)",
    },
    {
        "factor_column": "value_ps_score",
        "factor_name": "Sales-to-Price Style",
        "category": "Value",
        "direction": 1,
        "role": "alpha",
        "formula": "-log(positive PS TTM)",
    },
    {
        "factor_column": "value_pcf_score",
        "factor_name": "Cash-Flow Yield Style",
        "category": "Value",
        "direction": 1,
        "role": "alpha",
        "formula": "-log(positive PCF NCF TTM)",
    },
    {
        "factor_column": "log_float_market_cap",
        "factor_name": "Log Float Market Capitalization",
        "category": "Size",
        "direction": 0,
        "role": "neutralizer",
        "formula": "rolling_median(log(volume / turnover_fraction * close), 20)",
    },
]


FACTOR_COLUMNS = [
    item["factor_column"]
    for item in FACTOR_METADATA
]


def resolve_database_path() -> Path:
    """Find the SQLite database configured by the project."""

    try:
        from config import settings

        possible_names = [
            "DATABASE_PATH",
            "DB_PATH",
            "DATABASE_FILE",
            "DB_FILE",
        ]

        for name in possible_names:
            value = getattr(settings, name, None)

            if value:
                path = Path(value)

                if not path.is_absolute():
                    path = PROJECT_ROOT / path

                return path.resolve()

    except (ImportError, AttributeError):
        pass

    candidates = sorted(
        list((PROJECT_ROOT / "data").glob("*.db"))
        + list((PROJECT_ROOT / "data").glob("*.sqlite"))
        + list((PROJECT_ROOT / "data").glob("*.sqlite3"))
    )

    if len(candidates) == 1:
        return candidates[0].resolve()

    if len(candidates) == 0:
        raise FileNotFoundError(
            "No SQLite database was found. Check config/settings.py."
        )

    raise RuntimeError(
        "Multiple SQLite databases were found. "
        "Please define DATABASE_PATH in config/settings.py."
    )


def table_exists(
    connection: sqlite3.Connection,
    table_name: str,
) -> bool:
    query = """
    SELECT COUNT(*)
    FROM sqlite_master
    WHERE type = 'table'
      AND name = ?;
    """

    result = connection.execute(
        query,
        (table_name,),
    ).fetchone()

    return bool(result and result[0] > 0)


def get_table_columns(
    connection: sqlite3.Connection,
    table_name: str,
) -> list[str]:
    rows = connection.execute(
        f'PRAGMA table_info("{table_name}")'
    ).fetchall()

    return [row[1] for row in rows]


def select_and_rename_columns(
    connection: sqlite3.Connection,
    table_name: str,
    aliases: dict[str, list[str]],
    required_columns: set[str],
) -> pd.DataFrame:
    available_columns = set(
        get_table_columns(connection, table_name)
    )

    selected_columns: list[str] = []
    rename_map: dict[str, str] = {}

    for canonical_name, candidates in aliases.items():
        source_name = next(
            (
                candidate
                for candidate in candidates
                if candidate in available_columns
            ),
            None,
        )

        if source_name is not None:
            selected_columns.append(source_name)
            rename_map[source_name] = canonical_name

    missing_required = required_columns.difference(
        rename_map.values()
    )

    if missing_required:
        raise ValueError(
            f"Table {table_name} is missing required columns: "
            f"{sorted(missing_required)}"
        )

    quoted_columns = ", ".join(
        f'"{column}"'
        for column in selected_columns
    )

    data = pd.read_sql_query(
        f'SELECT {quoted_columns} FROM "{table_name}"',
        connection,
    )

    return data.rename(columns=rename_map)


def load_daily_prices(
    connection: sqlite3.Connection,
) -> pd.DataFrame:
    aliases = {
        "date": ["date", "trade_date"],
        "baostock_code": [
            "baostock_code",
            "code",
            "stock_code",
        ],
        "symbol": ["symbol", "yahoo_symbol"],
        "stock_name": ["stock_name", "name"],
        "close": ["close", "close_price"],
        "volume": ["volume", "vol"],
        "amount": ["amount", "trading_amount"],
        "turnover_rate": [
            "turnover_rate",
            "turn",
            "turnover",
        ],
        "trade_status": [
            "trade_status",
            "tradestatus",
            "is_trading",
        ],
        "pe_ttm": ["pe_ttm", "peTTM"],
        "pb_mrq": ["pb_mrq", "pbMRQ"],
        "ps_ttm": ["ps_ttm", "psTTM"],
        "pcf_ncf_ttm": [
            "pcf_ncf_ttm",
            "pcfNcfTTM",
        ],
    }

    prices = select_and_rename_columns(
        connection=connection,
        table_name=DAILY_TABLE,
        aliases=aliases,
        required_columns={
            "date",
            "baostock_code",
            "close",
        },
    )

    optional_numeric_columns = [
        "volume",
        "amount",
        "turnover_rate",
        "pe_ttm",
        "pb_mrq",
        "ps_ttm",
        "pcf_ncf_ttm",
    ]

    for column in optional_numeric_columns:
        if column not in prices.columns:
            prices[column] = np.nan

    if "trade_status" not in prices.columns:
        prices["trade_status"] = 1

    prices["date"] = pd.to_datetime(
        prices["date"],
        errors="coerce",
    )

    numeric_columns = [
        "close",
        "volume",
        "amount",
        "turnover_rate",
        "trade_status",
        "pe_ttm",
        "pb_mrq",
        "ps_ttm",
        "pcf_ncf_ttm",
    ]

    for column in numeric_columns:
        prices[column] = pd.to_numeric(
            prices[column],
            errors="coerce",
        )

    prices = prices.dropna(
        subset=["date", "baostock_code", "close"]
    )

    prices = prices.loc[
        (prices["trade_status"] == 1)
        & (prices["close"] > 0)
    ].copy()

    prices = prices.sort_values(
        ["baostock_code", "date"]
    ).reset_index(drop=True)

    duplicate_count = int(
        prices.duplicated(
            ["baostock_code", "date"]
        ).sum()
    )

    if duplicate_count:
        raise ValueError(
            f"{DAILY_TABLE} contains "
            f"{duplicate_count:,} duplicate stock-date rows."
        )

    return prices


def load_eligible_universe(
    connection: sqlite3.Connection,
) -> pd.DataFrame:
    aliases = {
        "snapshot_date": [
            "snapshot_date",
            "rebalance_date",
        ],
        "signal_price_date": [
            "signal_price_date",
            "price_date",
            "last_trade_date",
        ],
        "baostock_code": [
            "baostock_code",
            "code",
            "stock_code",
        ],
        "symbol": ["symbol", "yahoo_symbol"],
        "stock_name": ["stock_name", "name"],
        "industry": [
            "industry",
            "industry_name",
        ],
        "eligible": [
            "eligible",
            "is_eligible",
        ],
        "exclusion_reason": [
            "exclusion_reason",
            "reason",
        ],
    }

    universe = select_and_rename_columns(
        connection=connection,
        table_name=ELIGIBLE_TABLE,
        aliases=aliases,
        required_columns={
            "snapshot_date",
            "signal_price_date",
            "baostock_code",
        },
    )

    if "eligible" in universe.columns:
        eligible_text = (
            universe["eligible"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        universe = universe.loc[
            eligible_text.isin(
                ["1", "1.0", "true", "yes"]
            )
        ].copy()

    elif "exclusion_reason" in universe.columns:
        universe = universe.loc[
            universe["exclusion_reason"]
            .astype(str)
            .str.upper()
            .eq("ELIGIBLE")
        ].copy()

    else:
        raise ValueError(
            f"{ELIGIBLE_TABLE} must contain either "
            "'eligible' or 'exclusion_reason'."
        )

    universe["snapshot_date"] = pd.to_datetime(
        universe["snapshot_date"],
        errors="coerce",
    )

    universe["signal_price_date"] = pd.to_datetime(
        universe["signal_price_date"],
        errors="coerce",
    )

    universe = universe.dropna(
        subset=[
            "snapshot_date",
            "signal_price_date",
            "baostock_code",
        ]
    )

    universe = universe.drop_duplicates(
        ["snapshot_date", "baostock_code"],
        keep="last",
    )

    return universe


def add_master_information(
    connection: sqlite3.Connection,
    universe: pd.DataFrame,
) -> pd.DataFrame:
    required_information = {
        "symbol",
        "stock_name",
        "industry",
    }

    needs_master = any(
        column not in universe.columns
        or universe[column].isna().all()
        for column in required_information
    )

    if not needs_master:
        universe["industry"] = (
            universe["industry"]
            .fillna("UNKNOWN")
            .replace("", "UNKNOWN")
        )
        return universe

    if not table_exists(connection, MASTER_TABLE):
        for column in required_information:
            if column not in universe.columns:
                universe[column] = np.nan

        universe["industry"] = (
            universe["industry"]
            .fillna("UNKNOWN")
            .replace("", "UNKNOWN")
        )

        return universe

    aliases = {
        "baostock_code": [
            "baostock_code",
            "code",
            "stock_code",
        ],
        "master_symbol": [
            "symbol",
            "yahoo_symbol",
        ],
        "master_stock_name": [
            "stock_name",
            "name",
        ],
        "master_industry": [
            "industry",
            "industry_name",
        ],
    }

    master = select_and_rename_columns(
        connection=connection,
        table_name=MASTER_TABLE,
        aliases=aliases,
        required_columns={"baostock_code"},
    )

    master = master.drop_duplicates(
        "baostock_code",
        keep="last",
    )

    universe = universe.merge(
        master,
        on="baostock_code",
        how="left",
        validate="many_to_one",
    )

    information_pairs = [
        ("symbol", "master_symbol"),
        ("stock_name", "master_stock_name"),
        ("industry", "master_industry"),
    ]

    for target_column, master_column in information_pairs:
        if target_column not in universe.columns:
            universe[target_column] = np.nan

        if master_column in universe.columns:
            universe[target_column] = universe[
                target_column
            ].combine_first(universe[master_column])

            universe = universe.drop(
                columns=[master_column]
            )

    universe["industry"] = (
        universe["industry"]
        .fillna("UNKNOWN")
        .replace("", "UNKNOWN")
    )

    return universe


def grouped_rolling(
    values: pd.Series,
    groups: pd.Series,
    window: int,
    min_periods: int,
    operation: str,
) -> pd.Series:
    rolling_object = values.groupby(
        groups,
        sort=False,
    ).rolling(
        window=window,
        min_periods=min_periods,
    )

    if operation == "mean":
        result = rolling_object.mean()
    elif operation == "std":
        result = rolling_object.std()
    elif operation == "max":
        result = rolling_object.max()
    elif operation == "median":
        result = rolling_object.median()
    else:
        raise ValueError(
            f"Unsupported rolling operation: {operation}"
        )

    return (
        result
        .reset_index(level=0, drop=True)
        .sort_index()
    )


def positive_negative_log(
    values: pd.Series,
) -> pd.Series:
    valid_values = values.where(values > 0)

    return -np.log(valid_values)


def calculate_raw_factors(
    prices: pd.DataFrame,
) -> pd.DataFrame:
    prices = prices.copy()

    stock_group = prices.groupby(
        "baostock_code",
        sort=False,
    )

    close_lag_1 = stock_group["close"].shift(1)
    close_lag_5 = stock_group["close"].shift(5)
    close_lag_20 = stock_group["close"].shift(20)
    close_lag_60 = stock_group["close"].shift(60)
    close_lag_120 = stock_group["close"].shift(120)
    close_lag_252 = stock_group["close"].shift(252)

    prices["return_1d"] = (
        prices["close"] / close_lag_1 - 1.0
    )

    prices["momentum_20"] = (
        prices["close"] / close_lag_20 - 1.0
    )

    prices["momentum_60"] = (
        prices["close"] / close_lag_60 - 1.0
    )

    prices["momentum_120"] = (
        prices["close"] / close_lag_120 - 1.0
    )

    prices["momentum_12_1"] = (
        close_lag_20 / close_lag_252 - 1.0
    )

    prices["reversal_5_score"] = -(
        prices["close"] / close_lag_5 - 1.0
    )

    prices["volatility_20"] = grouped_rolling(
        values=prices["return_1d"],
        groups=prices["baostock_code"],
        window=20,
        min_periods=15,
        operation="std",
    ) * math.sqrt(252)

    negative_returns = prices["return_1d"].clip(
        upper=0
    )

    prices["downside_volatility_60"] = grouped_rolling(
        values=negative_returns,
        groups=prices["baostock_code"],
        window=60,
        min_periods=40,
        operation="std",
    ) * math.sqrt(252)

    rolling_high_60 = grouped_rolling(
        values=prices["close"],
        groups=prices["baostock_code"],
        window=60,
        min_periods=40,
        operation="max",
    )

    prices["drawdown_60"] = (
        prices["close"] / rolling_high_60 - 1.0
    )

    moving_average_60 = grouped_rolling(
        values=prices["close"],
        groups=prices["baostock_code"],
        window=60,
        min_periods=40,
        operation="mean",
    )

    prices["trend_60"] = (
        prices["close"] / moving_average_60 - 1.0
    )

    average_amount_20 = grouped_rolling(
        values=prices["amount"],
        groups=prices["baostock_code"],
        window=20,
        min_periods=10,
        operation="mean",
    )

    prices["liquidity_amount_20"] = np.log1p(
        average_amount_20 / 1_000_000
    )

    prices["turnover_20"] = grouped_rolling(
        values=prices["turnover_rate"],
        groups=prices["baostock_code"],
        window=20,
        min_periods=10,
        operation="mean",
    )

    turnover_std_20 = grouped_rolling(
        values=prices["turnover_rate"],
        groups=prices["baostock_code"],
        window=20,
        min_periods=10,
        operation="std",
    )

    prices["turnover_stability_20"] = (
        -turnover_std_20
    )

    amount_in_100_million = (
        prices["amount"] / 100_000_000
    )

    daily_amihud = (
        prices["return_1d"].abs()
        / amount_in_100_million.where(
            amount_in_100_million > 0
        )
    )

    amihud_20 = grouped_rolling(
        values=daily_amihud,
        groups=prices["baostock_code"],
        window=20,
        min_periods=10,
        operation="mean",
    )

    prices["amihud_liquidity_score_20"] = (
        -np.log1p(amihud_20)
    )

    prices["value_pe_score"] = positive_negative_log(
        prices["pe_ttm"]
    )

    prices["value_pb_score"] = positive_negative_log(
        prices["pb_mrq"]
    )

    prices["value_ps_score"] = positive_negative_log(
        prices["ps_ttm"]
    )

    prices["value_pcf_score"] = positive_negative_log(
        prices["pcf_ncf_ttm"]
    )

    turnover_fraction = (
        prices["turnover_rate"] / 100.0
    )

    estimated_float_shares = (
        prices["volume"]
        / turnover_fraction.where(
            turnover_fraction > 0.0001
        )
    )

    estimated_float_market_cap = (
        estimated_float_shares
        * prices["close"]
    )

    daily_log_float_market_cap = np.log(
        estimated_float_market_cap.where(
            estimated_float_market_cap > 0
        )
    )

    prices["log_float_market_cap"] = grouped_rolling(
        values=daily_log_float_market_cap,
        groups=prices["baostock_code"],
        window=20,
        min_periods=5,
        operation="median",
    )

    prices[FACTOR_COLUMNS] = prices[
        FACTOR_COLUMNS
    ].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return prices


def build_monthly_factor_panel(
    universe: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    selected_price_columns = [
        "date",
        "baostock_code",
        "close",
        *FACTOR_COLUMNS,
    ]

    monthly_factors = universe.merge(
        prices[selected_price_columns],
        left_on=[
            "baostock_code",
            "signal_price_date",
        ],
        right_on=[
            "baostock_code",
            "date",
        ],
        how="left",
        validate="one_to_one",
    )

    monthly_factors = monthly_factors.drop(
        columns=["date"],
        errors="ignore",
    )

    preferred_columns = [
        "snapshot_date",
        "signal_price_date",
        "baostock_code",
        "symbol",
        "stock_name",
        "industry",
        "close",
        *FACTOR_COLUMNS,
    ]

    for column in preferred_columns:
        if column not in monthly_factors.columns:
            monthly_factors[column] = np.nan

    monthly_factors = monthly_factors[
        preferred_columns
    ].sort_values(
        ["snapshot_date", "baostock_code"]
    ).reset_index(drop=True)

    return monthly_factors


def build_coverage_summary(
    factor_panel: pd.DataFrame,
) -> pd.DataFrame:
    metadata = pd.DataFrame(FACTOR_METADATA)

    coverage_rows: list[dict] = []

    number_of_rows = len(factor_panel)

    for factor_column in FACTOR_COLUMNS:
        valid_mask = factor_panel[
            factor_column
        ].notna()

        non_missing = int(valid_mask.sum())

        if number_of_rows:
            coverage_percent = (
                non_missing / number_of_rows * 100
            )
        else:
            coverage_percent = 0.0

        valid_dates = factor_panel.loc[
            valid_mask,
            "snapshot_date",
        ]

        if valid_dates.empty:
            first_valid_snapshot = None
            last_valid_snapshot = None
        else:
            first_valid_snapshot = (
                valid_dates.min().date().isoformat()
            )
            last_valid_snapshot = (
                valid_dates.max().date().isoformat()
            )

        coverage_rows.append(
            {
                "factor_column": factor_column,
                "non_missing_observations": non_missing,
                "total_observations": number_of_rows,
                "coverage_percent": coverage_percent,
                "first_valid_snapshot": first_valid_snapshot,
                "last_valid_snapshot": last_valid_snapshot,
            }
        )

    coverage = pd.DataFrame(coverage_rows)

    coverage = metadata.merge(
        coverage,
        on="factor_column",
        how="left",
        validate="one_to_one",
    )

    return coverage


def save_results(
    connection: sqlite3.Connection,
    factor_panel: pd.DataFrame,
    coverage: pd.DataFrame,
) -> None:
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    database_output = factor_panel.copy()

    for column in [
        "snapshot_date",
        "signal_price_date",
    ]:
        database_output[column] = (
            pd.to_datetime(
                database_output[column]
            )
            .dt.strftime("%Y-%m-%d")
        )

    database_output.to_sql(
        OUTPUT_TABLE,
        connection,
        if_exists="replace",
        index=False,
        chunksize=5_000,
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

    csv_output = database_output.copy()

    csv_output.to_csv(
        RAW_FACTOR_FILE,
        index=False,
        encoding="utf-8-sig",
        float_format="%.10f",
    )

    coverage.to_csv(
        COVERAGE_FILE,
        index=False,
        encoding="utf-8-sig",
        float_format="%.6f",
    )

    pd.DataFrame(FACTOR_METADATA).to_csv(
        METADATA_FILE,
        index=False,
        encoding="utf-8-sig",
    )


def main() -> None:
    pd.set_option(
        "display.max_columns",
        None,
    )
    pd.set_option(
        "display.width",
        180,
    )

    database_path = resolve_database_path()

    print("=" * 110)
    print("V2 RAW FACTOR LIBRARY")
    print("=" * 110)
    print(f"Database: {database_path}")
    print()

    with sqlite3.connect(database_path) as connection:
        for table_name in [
            DAILY_TABLE,
            ELIGIBLE_TABLE,
        ]:
            if not table_exists(
                connection,
                table_name,
            ):
                raise RuntimeError(
                    f"Required table does not exist: "
                    f"{table_name}"
                )

        print("Loading daily market data...", flush=True)
        prices = load_daily_prices(connection)

        print(
            f"Daily rows loaded: {len(prices):,}",
            flush=True,
        )
        print(
            f"Stocks loaded: "
            f"{prices['baostock_code'].nunique():,}",
            flush=True,
        )

        print(
            "Loading point-in-time eligible universe...",
            flush=True,
        )
        universe = load_eligible_universe(connection)

        universe = add_master_information(
            connection,
            universe,
        )

        print(
            f"Eligible observations: "
            f"{len(universe):,}",
            flush=True,
        )

        print(
            "Calculating raw daily factors...",
            flush=True,
        )
        prices = calculate_raw_factors(prices)

        print(
            "Building monthly factor panel...",
            flush=True,
        )
        factor_panel = build_monthly_factor_panel(
            universe=universe,
            prices=prices,
        )

        coverage = build_coverage_summary(
            factor_panel
        )

        print(
            "Saving factor library...",
            flush=True,
        )
        save_results(
            connection=connection,
            factor_panel=factor_panel,
            coverage=coverage,
        )

    print()
    print("=" * 110)
    print("V2 FACTOR-LIBRARY SUMMARY")
    print("=" * 110)

    print(
        f"Monthly factor observations: "
        f"{len(factor_panel):,}"
    )
    print(
        f"Snapshots: "
        f"{factor_panel['snapshot_date'].nunique():,}"
    )
    print(
        f"Historical stocks: "
        f"{factor_panel['baostock_code'].nunique():,}"
    )
    print(
        f"Raw factors and neutralizers: "
        f"{len(FACTOR_COLUMNS)}"
    )

    print()
    print("=" * 110)
    print("FACTOR COVERAGE")
    print("=" * 110)

    display_columns = [
        "factor_column",
        "factor_name",
        "category",
        "role",
        "coverage_percent",
        "first_valid_snapshot",
    ]

    print(
        coverage[display_columns].to_string(
            index=False,
            formatters={
                "coverage_percent": lambda value: (
                    f"{value:.2f}%"
                )
            },
        )
    )

    print()
    print(f"Database table: {OUTPUT_TABLE}")
    print(f"Raw-factor CSV: {RAW_FACTOR_FILE}")
    print(f"Coverage CSV: {COVERAGE_FILE}")
    print(f"Metadata CSV: {METADATA_FILE}")
    print()
    print("V2 raw factor library completed.")


if __name__ == "__main__":
    main()