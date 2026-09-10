"""
V2 point-in-time tradable universe.

For every historical CSI 300 snapshot, determine which stocks
were actually eligible for portfolio selection at that time.
"""

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import DATABASE_PATH


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = PROJECT_ROOT / "reports" / "results"

ELIGIBILITY_FILE = (
    RESULT_DIR / "tradable_universe_v2.csv"
)
SUMMARY_FILE = (
    RESULT_DIR / "tradability_summary.csv"
)

MIN_TRADING_AGE = 120
MIN_AVERAGE_AMOUNT_20 = 10_000_000
MAX_STALE_CALENDAR_DAYS = 14
RESUMPTION_GAP_DAYS = 30
RESUMPTION_COOLDOWN_DAYS = 5

OUTPUT_COLUMNS = [
    "snapshot_date",
    "signal_price_date",
    "baostock_code",
    "symbol",
    "stock_name",
    "industry",
    "close",
    "average_amount_20",
    "trading_age",
    "days_since_last_trade",
    "days_after_resumption",
    "is_st",
    "out_date",
    "eligible",
    "exclusion_reason",
]


def load_membership(connection):
    query = """
        SELECT
            query_date,
            baostock_code,
            symbol,
            stock_name
        FROM index_constituents_history
        WHERE index_code = '000300'
        ORDER BY
            query_date,
            baostock_code;
    """

    membership = pd.read_sql_query(
        query,
        connection,
    )

    if membership.empty:
        raise ValueError(
            "Historical membership table is empty."
        )

    membership["query_date"] = pd.to_datetime(
        membership["query_date"]
    )

    return membership


def load_stock_master(connection):
    query = """
        SELECT
            baostock_code,
            ipo_date,
            out_date,
            industry
        FROM stock_master_v2;
    """

    master = pd.read_sql_query(
        query,
        connection,
    )

    if master.empty:
        raise ValueError(
            "stock_master_v2 is empty."
        )

    master["ipo_date"] = pd.to_datetime(
        master["ipo_date"],
        errors="coerce",
    )

    master["out_date"] = pd.to_datetime(
        master["out_date"],
        errors="coerce",
    )

    return master


def load_and_prepare_prices(connection):
    query = """
        SELECT
            date,
            baostock_code,
            close,
            amount,
            trade_status,
            pct_change,
            is_st
        FROM daily_prices_v2
        ORDER BY
            baostock_code,
            date;
    """

    prices = pd.read_sql_query(
        query,
        connection,
    )

    if prices.empty:
        raise ValueError(
            "daily_prices_v2 is empty."
        )

    prices["date"] = pd.to_datetime(
        prices["date"]
    )

    numeric_columns = [
        "close",
        "amount",
        "trade_status",
        "pct_change",
        "is_st",
    ]

    for column in numeric_columns:
        prices[column] = pd.to_numeric(
            prices[column],
            errors="coerce",
        )

    # Keep only days on which the stock traded.
    prices = prices[
        prices["trade_status"] == 1
    ].copy()

    prices = prices.sort_values(
        ["baostock_code", "date"]
    ).reset_index(drop=True)

    prices["trading_age"] = (
        prices.groupby(
            "baostock_code",
            sort=False,
        )
        .cumcount()
        .add(1)
    )

    prices["average_amount_20"] = (
        prices.groupby(
            "baostock_code",
            sort=False,
        )["amount"]
        .transform(
            lambda values: values.rolling(
                window=20,
                min_periods=20,
            ).mean()
        )
    )

    prices["previous_trade_date"] = (
        prices.groupby(
            "baostock_code",
            sort=False,
        )["date"]
        .shift(1)
    )

    prices["gap_days"] = (
        prices["date"]
        - prices["previous_trade_date"]
    ).dt.days

    prices["resumption_event"] = (
        prices["gap_days"]
        > RESUMPTION_GAP_DAYS
    )

    prices["resumption_number"] = (
        prices.groupby(
            "baostock_code",
            sort=False,
        )["resumption_event"]
        .cumsum()
    )

    prices["days_after_resumption"] = (
        prices.groupby(
            [
                "baostock_code",
                "resumption_number",
            ],
            sort=False,
        )
        .cumcount()
    )

    prices.loc[
        prices["resumption_number"] == 0,
        "days_after_resumption",
    ] = 999

    return prices[
        [
            "date",
            "baostock_code",
            "close",
            "average_amount_20",
            "trading_age",
            "days_after_resumption",
            "is_st",
        ]
    ]


def combine_membership_and_prices(
    membership,
    prices,
):
    left = membership.sort_values(
        ["query_date", "baostock_code"]
    ).reset_index(drop=True)

    right = prices.sort_values(
        ["date", "baostock_code"]
    ).reset_index(drop=True)

    combined = pd.merge_asof(
        left,
        right,
        left_on="query_date",
        right_on="date",
        by="baostock_code",
        direction="backward",
        allow_exact_matches=True,
    )

    combined = combined.rename(
        columns={
            "query_date": "snapshot_date",
            "date": "signal_price_date",
        }
    )

    combined["days_since_last_trade"] = (
        combined["snapshot_date"]
        - combined["signal_price_date"]
    ).dt.days

    return combined


def assign_eligibility(frame):
    has_price = (
        frame["signal_price_date"].notna()
        & frame["close"].gt(0)
    )

    listed_long_enough = (
        frame["trading_age"]
        >= MIN_TRADING_AGE
    )

    recently_traded = (
        frame["days_since_last_trade"]
        <= MAX_STALE_CALENDAR_DAYS
    )

    not_st = frame["is_st"].eq(0)

    not_delisted = (
        frame["out_date"].isna()
        | (
            frame["snapshot_date"]
            < frame["out_date"]
        )
    )

    liquid_enough = (
        frame["average_amount_20"]
        >= MIN_AVERAGE_AMOUNT_20
    )

    outside_resumption_cooldown = (
        frame["days_after_resumption"]
        >= RESUMPTION_COOLDOWN_DAYS
    )

    frame["eligible"] = (
        has_price
        & listed_long_enough
        & recently_traded
        & not_st
        & not_delisted
        & liquid_enough
        & outside_resumption_cooldown
    ).astype(int)

    conditions = [
        ~has_price,
        ~not_delisted,
        ~listed_long_enough,
        ~recently_traded,
        ~not_st,
        ~outside_resumption_cooldown,
        ~liquid_enough,
    ]

    reasons = [
        "NO_VALID_PRICE",
        "DELISTED",
        "INSUFFICIENT_HISTORY",
        "SUSPENDED_OR_STALE",
        "ST_STOCK",
        "RESUMPTION_COOLDOWN",
        "LOW_LIQUIDITY",
    ]

    frame["exclusion_reason"] = np.select(
        conditions,
        reasons,
        default="ELIGIBLE",
    )

    return frame


def build_summary(eligibility):
    basic_summary = (
        eligibility.groupby("snapshot_date")
        .agg(
            index_constituents=(
                "baostock_code",
                "count",
            ),
            eligible_stocks=(
                "eligible",
                "sum",
            ),
        )
        .reset_index()
    )

    basic_summary["excluded_stocks"] = (
        basic_summary["index_constituents"]
        - basic_summary["eligible_stocks"]
    )

    basic_summary["eligible_percent"] = (
        basic_summary["eligible_stocks"]
        / basic_summary["index_constituents"]
        * 100
    )

    reason_counts = pd.crosstab(
        eligibility["snapshot_date"],
        eligibility["exclusion_reason"],
    ).reset_index()

    summary = basic_summary.merge(
        reason_counts,
        on="snapshot_date",
        how="left",
    )

    return summary.sort_values(
        "snapshot_date"
    ).reset_index(drop=True)


def create_eligibility_table(connection):
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS
        eligible_universe_v2 (
            snapshot_date TEXT NOT NULL,
            signal_price_date TEXT,
            baostock_code TEXT NOT NULL,
            symbol TEXT NOT NULL,
            stock_name TEXT,
            industry TEXT,
            close REAL,
            average_amount_20 REAL,
            trading_age INTEGER,
            days_since_last_trade INTEGER,
            days_after_resumption INTEGER,
            is_st INTEGER,
            out_date TEXT,
            eligible INTEGER NOT NULL,
            exclusion_reason TEXT NOT NULL,
            PRIMARY KEY (
                snapshot_date,
                baostock_code
            )
        );
        """
    )

    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS
        idx_eligible_universe_v2_date
        ON eligible_universe_v2 (
            snapshot_date,
            eligible
        );
        """
    )

    connection.commit()


def save_to_database(
    connection,
    eligibility,
):
    create_eligibility_table(connection)

    connection.execute(
        "DELETE FROM eligible_universe_v2;"
    )

    database_frame = eligibility[
        OUTPUT_COLUMNS
    ].copy()

    date_columns = [
        "snapshot_date",
        "signal_price_date",
        "out_date",
    ]

    for column in date_columns:
        database_frame[column] = (
            database_frame[column]
            .dt.strftime("%Y-%m-%d")
        )

    clean_frame = (
        database_frame.astype(object)
        .where(
            pd.notna(database_frame),
            None,
        )
    )

    records = list(
        clean_frame.itertuples(
            index=False,
            name=None,
        )
    )

    placeholders = ", ".join(
        ["?"] * len(OUTPUT_COLUMNS)
    )

    column_names = ", ".join(
        OUTPUT_COLUMNS
    )

    connection.executemany(
        f"""
        INSERT INTO eligible_universe_v2 (
            {column_names}
        )
        VALUES ({placeholders});
        """,
        records,
    )

    connection.commit()


def main():
    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading historical membership...")

    with sqlite3.connect(DATABASE_PATH) as connection:
        membership = load_membership(connection)
        master = load_stock_master(connection)

        print("Preparing historical price features...")
        prices = load_and_prepare_prices(
            connection
        )

        print("Matching each snapshot to market data...")
        eligibility = (
            combine_membership_and_prices(
                membership,
                prices,
            )
        )

        eligibility = eligibility.merge(
            master,
            on="baostock_code",
            how="left",
        )

        eligibility = assign_eligibility(
            eligibility
        )

        summary = build_summary(
            eligibility
        )

        save_to_database(
            connection,
            eligibility,
        )

    export_frame = eligibility[
        OUTPUT_COLUMNS
    ].copy()

    for column in [
        "snapshot_date",
        "signal_price_date",
        "out_date",
    ]:
        export_frame[column] = (
            export_frame[column]
            .dt.strftime("%Y-%m-%d")
        )

    export_frame.to_csv(
        ELIGIBILITY_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    summary_export = summary.copy()

    summary_export["snapshot_date"] = (
        summary_export["snapshot_date"]
        .dt.strftime("%Y-%m-%d")
    )

    summary_export.to_csv(
        SUMMARY_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    usable_snapshots = summary[
        summary["eligible_stocks"] >= 100
    ]

    first_usable_date = (
        usable_snapshots["snapshot_date"].min()
        if not usable_snapshots.empty
        else pd.NaT
    )

    reason_summary = (
        eligibility["exclusion_reason"]
        .value_counts()
        .rename_axis("reason")
        .reset_index(name="observations")
    )

    print("\n" + "=" * 110)
    print("V2 TRADABILITY SUMMARY")
    print("=" * 110)
    print(
        f"Historical snapshots: "
        f"{summary['snapshot_date'].nunique()}"
    )
    print(
        f"Membership observations: "
        f"{len(eligibility):,}"
    )
    print(
        f"First usable snapshot: "
        f"{first_usable_date:%Y-%m-%d}"
        if pd.notna(first_usable_date)
        else "First usable snapshot: None"
    )
    print(
        f"Latest eligible stocks: "
        f"{int(summary.iloc[-1]['eligible_stocks'])}"
    )
    print(
        f"Average eligible stocks: "
        f"{summary['eligible_stocks'].mean():.1f}"
    )

    print("\nExclusion-reason totals:")
    print(
        reason_summary.to_string(index=False)
    )

    print("\nLatest 10 snapshots:")
    print(
        summary[
            [
                "snapshot_date",
                "index_constituents",
                "eligible_stocks",
                "excluded_stocks",
                "eligible_percent",
            ]
        ]
        .tail(10)
        .to_string(
            index=False,
            formatters={
                "eligible_percent":
                    lambda value: f"{value:.2f}%"
            },
        )
    )

    print(f"\nDatabase table: eligible_universe_v2")
    print(f"Eligibility CSV: {ELIGIBILITY_FILE}")
    print(f"Summary CSV: {SUMMARY_FILE}")
    print("\nV2 tradability analysis completed.")


if __name__ == "__main__":
    main()