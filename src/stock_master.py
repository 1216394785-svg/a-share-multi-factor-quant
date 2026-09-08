"""
V2 stock master database.

Combines:
- historical CSI 300 membership
- IPO and delisting dates
- listing status
- industry classifications
"""

import sqlite3
import time
from datetime import datetime
from pathlib import Path

import baostock as bs
import pandas as pd

from config.settings import DATABASE_PATH
from src.baostock_loader import (
    close_baostock_connection,
    connect_baostock,
)
from src.universe import result_set_to_dataframe


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = PROJECT_ROOT / "reports" / "results"

MASTER_FILE = RESULT_DIR / "stock_master_v2.csv"
INACTIVE_FILE = RESULT_DIR / "inactive_stock_review.csv"

MASTER_COLUMNS = [
    "baostock_code",
    "symbol",
    "stock_name",
    "ipo_date",
    "out_date",
    "security_type",
    "listing_status",
    "industry",
    "industry_classification",
    "industry_update_date",
    "first_hs300_snapshot",
    "last_hs300_snapshot",
    "is_current_hs300_member",
    "source",
    "updated_at",
]


def run_baostock_query(
    query_function,
    query_name,
    max_retries=3,
    timeout_seconds=60,
):
    final_error = ""

    for attempt in range(1, max_retries + 1):
        try:
            connect_baostock(timeout_seconds)

            print(
                f"Downloading {query_name} "
                f"(attempt {attempt}/{max_retries})..."
            )

            result = query_function()
            frame = result_set_to_dataframe(result)

            if frame.empty:
                raise ValueError(
                    f"No data returned for {query_name}."
                )

            close_baostock_connection()

            return frame

        except Exception as error:
            final_error = str(error)
            close_baostock_connection()

            print(
                f"Attempt {attempt} failed: {error}"
            )

            if attempt < max_retries:
                time.sleep(5 * attempt)

    raise RuntimeError(
        f"Unable to download {query_name}: "
        f"{final_error}"
    )


def load_historical_membership(connection):
    query = """
        WITH latest_snapshot AS (
            SELECT MAX(query_date) AS latest_date
            FROM index_constituents_history
            WHERE index_code = '000300'
        ),
        ranked_names AS (
            SELECT
                baostock_code,
                symbol,
                stock_name,
                query_date,
                ROW_NUMBER() OVER (
                    PARTITION BY baostock_code
                    ORDER BY query_date DESC
                ) AS row_number
            FROM index_constituents_history
            WHERE index_code = '000300'
        ),
        membership_dates AS (
            SELECT
                baostock_code,
                MIN(query_date)
                    AS first_hs300_snapshot,
                MAX(query_date)
                    AS last_hs300_snapshot
            FROM index_constituents_history
            WHERE index_code = '000300'
            GROUP BY baostock_code
        )
        SELECT
            names.baostock_code,
            names.symbol,
            names.stock_name
                AS historical_stock_name,
            dates.first_hs300_snapshot,
            dates.last_hs300_snapshot,
            CASE
                WHEN dates.last_hs300_snapshot
                     = latest.latest_date
                THEN 1
                ELSE 0
            END AS is_current_hs300_member
        FROM ranked_names AS names
        JOIN membership_dates AS dates
          ON names.baostock_code
             = dates.baostock_code
        CROSS JOIN latest_snapshot AS latest
        WHERE names.row_number = 1
        ORDER BY names.baostock_code;
    """

    membership = pd.read_sql_query(
        query,
        connection,
    )

    if membership.empty:
        raise ValueError(
            "Historical membership data is empty."
        )

    return membership


def prepare_basic_information(basic):
    required_columns = [
        "code",
        "code_name",
        "ipoDate",
        "outDate",
        "type",
        "status",
    ]

    for column in required_columns:
        if column not in basic.columns:
            basic[column] = pd.NA

    basic = basic[required_columns].copy()

    basic = basic.rename(
        columns={
            "code": "baostock_code",
            "code_name": "basic_stock_name",
            "ipoDate": "ipo_date",
            "outDate": "out_date",
            "type": "security_type",
            "status": "listing_status",
        }
    )

    basic["ipo_date"] = basic[
        "ipo_date"
    ].replace("", pd.NA)

    basic["out_date"] = basic[
        "out_date"
    ].replace("", pd.NA)

    basic["security_type"] = pd.to_numeric(
        basic["security_type"],
        errors="coerce",
    )

    basic["listing_status"] = pd.to_numeric(
        basic["listing_status"],
        errors="coerce",
    )

    return basic.drop_duplicates(
        subset=["baostock_code"],
        keep="last",
    )


def prepare_industry_information(industry):
    required_columns = [
        "code",
        "industry",
        "industryClassification",
        "updateDate",
    ]

    for column in required_columns:
        if column not in industry.columns:
            industry[column] = pd.NA

    industry = industry[
        required_columns
    ].copy()

    industry = industry.rename(
        columns={
            "code": "baostock_code",
            "industryClassification":
                "industry_classification",
            "updateDate":
                "industry_update_date",
        }
    )

    industry["industry"] = industry[
        "industry"
    ].replace("", pd.NA)

    industry["industry_classification"] = (
        industry["industry_classification"]
        .replace("", pd.NA)
    )

    return industry.drop_duplicates(
        subset=["baostock_code"],
        keep="last",
    )


def build_stock_master(
    membership,
    basic,
    industry,
):
    master = membership.merge(
        basic,
        on="baostock_code",
        how="left",
    )

    master = master.merge(
        industry,
        on="baostock_code",
        how="left",
    )

    master["stock_name"] = master[
        "basic_stock_name"
    ].fillna(
        master["historical_stock_name"]
    )

    master["industry"] = master[
        "industry"
    ].fillna("Unknown")

    master["industry_classification"] = master[
        "industry_classification"
    ].fillna("Unknown")

    master["source"] = "baostock"

    master["updated_at"] = (
        datetime.now().isoformat(
            timespec="seconds"
        )
    )

    master = master[MASTER_COLUMNS]

    master = master.sort_values(
        "baostock_code"
    ).reset_index(drop=True)

    return master


def create_master_table(connection):
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS stock_master_v2 (
            baostock_code TEXT PRIMARY KEY,
            symbol TEXT NOT NULL,
            stock_name TEXT,
            ipo_date TEXT,
            out_date TEXT,
            security_type INTEGER,
            listing_status INTEGER,
            industry TEXT,
            industry_classification TEXT,
            industry_update_date TEXT,
            first_hs300_snapshot TEXT,
            last_hs300_snapshot TEXT,
            is_current_hs300_member INTEGER,
            source TEXT,
            updated_at TEXT
        );
        """
    )

    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS
        idx_stock_master_v2_industry
        ON stock_master_v2 (industry);
        """
    )

    connection.commit()


def save_master(connection, master):
    create_master_table(connection)

    connection.execute(
        "DELETE FROM stock_master_v2;"
    )

    clean_master = master.astype(object).where(
        pd.notna(master),
        None,
    )

    records = list(
        clean_master.itertuples(
            index=False,
            name=None,
        )
    )

    placeholders = ", ".join(
        ["?"] * len(MASTER_COLUMNS)
    )

    columns = ", ".join(MASTER_COLUMNS)

    connection.executemany(
        f"""
        INSERT INTO stock_master_v2 (
            {columns}
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

    basic_raw = run_baostock_query(
        bs.query_stock_basic,
        "stock basic information",
    )

    industry_raw = run_baostock_query(
        bs.query_stock_industry,
        "industry information",
    )

    basic = prepare_basic_information(
        basic_raw
    )

    industry = prepare_industry_information(
        industry_raw
    )

    with sqlite3.connect(DATABASE_PATH) as connection:
        membership = load_historical_membership(
            connection
        )

        master = build_stock_master(
            membership,
            basic,
            industry,
        )

        save_master(
            connection,
            master,
        )

    inactive_mask = (
        master["out_date"].notna()
        | (
            master["listing_status"]
            .fillna(-1)
            != 1
        )
    )

    inactive = master[
        inactive_mask
    ].copy()

    master.to_csv(
        MASTER_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    inactive.to_csv(
        INACTIVE_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print("=" * 110)
    print("V2 STOCK MASTER SUMMARY")
    print("=" * 110)
    print(f"Total historical stocks: {len(master)}")
    print(
        "Current CSI 300 members: "
        f"{int(master['is_current_hs300_member'].sum())}"
    )
    print(f"Inactive/delisted stocks: {len(inactive)}")
    print(
        "Missing IPO dates: "
        f"{int(master['ipo_date'].isna().sum())}"
    )
    print(
        "Unknown industries: "
        f"{int((master['industry'] == 'Unknown').sum())}"
    )

    if not inactive.empty:
        print("\nInactive/delisted stock review:")
        print(
            inactive[
                [
                    "baostock_code",
                    "stock_name",
                    "ipo_date",
                    "out_date",
                    "listing_status",
                    "last_hs300_snapshot",
                ]
            ].to_string(index=False)
        )

    print(f"\nDatabase table: stock_master_v2")
    print(f"Master CSV: {MASTER_FILE}")
    print(f"Inactive review: {INACTIVE_FILE}")
    print("\nV2 stock master completed.")


if __name__ == "__main__":
    main()