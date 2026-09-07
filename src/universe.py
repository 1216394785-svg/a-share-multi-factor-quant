"""
V2 stock universe module.

Download the current CSI 300 constituents and industry information
from BaoStock, then save the snapshot into SQLite and CSV.
"""

import sqlite3
from datetime import date
from pathlib import Path

import baostock as bs
import pandas as pd

from config.settings import DATABASE_PATH


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = PROJECT_ROOT / "reports" / "results"
OUTPUT_FILE = RESULT_DIR / "stock_universe_v2.csv"

INDEX_CODE = "000300"
INDEX_NAME = "CSI 300"


def result_set_to_dataframe(result_set):
    """Convert a BaoStock result set into a pandas DataFrame."""

    if result_set.error_code != "0":
        raise RuntimeError(
            f"BaoStock query failed: "
            f"{result_set.error_code} - {result_set.error_msg}"
        )

    rows = []

    while result_set.next():
        rows.append(result_set.get_row_data())

    return pd.DataFrame(rows, columns=result_set.fields)


def baostock_to_yahoo_symbol(baostock_code):
    """
    Convert BaoStock symbols into the format used by the existing project.

    Examples:
        sh.600519 -> 600519.SS
        sz.000001 -> 000001.SZ
    """

    exchange, stock_code = baostock_code.split(".")

    if exchange == "sh":
        return f"{stock_code}.SS"

    if exchange == "sz":
        return f"{stock_code}.SZ"

    if exchange == "bj":
        return f"{stock_code}.BJ"

    raise ValueError(f"Unsupported BaoStock code: {baostock_code}")


def download_hs300_universe():
    """Download CSI 300 constituents and industry information."""

    print("Connecting to BaoStock...")

    login_result = bs.login()

    if login_result.error_code != "0":
        raise RuntimeError(
            f"BaoStock login failed: "
            f"{login_result.error_code} - {login_result.error_msg}"
        )

    try:
        print("Downloading CSI 300 constituents...")

        constituent_result = bs.query_hs300_stocks()
        constituents = result_set_to_dataframe(constituent_result)

        if constituents.empty:
            raise ValueError("No CSI 300 constituent data was returned.")

        print("Downloading stock industry information...")

        industry_result = bs.query_stock_industry()
        industries = result_set_to_dataframe(industry_result)

    finally:
        bs.logout()

    if "updateDate" in constituents.columns:
        available_dates = constituents["updateDate"].replace("", pd.NA).dropna()

        if not available_dates.empty:
            snapshot_date = available_dates.max()
        else:
            snapshot_date = date.today().isoformat()
    else:
        snapshot_date = date.today().isoformat()

    constituent_columns = ["code", "code_name"]
    constituents = constituents[constituent_columns].copy()

    if industries.empty:
        industries = pd.DataFrame(
            columns=[
                "code",
                "industry",
                "industryClassification",
            ]
        )
    else:
        required_industry_columns = [
            "code",
            "industry",
            "industryClassification",
        ]

        for column in required_industry_columns:
            if column not in industries.columns:
                industries[column] = ""

        industries = industries[required_industry_columns].copy()
        industries = industries.drop_duplicates(
            subset=["code"],
            keep="last",
        )

    universe = constituents.merge(
        industries,
        on="code",
        how="left",
    )

    universe = universe.rename(
        columns={
            "code": "baostock_code",
            "code_name": "stock_name",
            "industryClassification": "industry_classification",
        }
    )

    universe["symbol"] = universe["baostock_code"].apply(
        baostock_to_yahoo_symbol
    )

    universe["exchange"] = universe["baostock_code"].str[:2].str.upper()
    universe["snapshot_date"] = snapshot_date
    universe["index_code"] = INDEX_CODE
    universe["index_name"] = INDEX_NAME
    universe["is_active"] = 1

    universe["industry"] = universe["industry"].fillna("Unknown")
    universe["industry_classification"] = (
        universe["industry_classification"].fillna("Unknown")
    )

    universe = universe[
        [
            "snapshot_date",
            "index_code",
            "index_name",
            "baostock_code",
            "symbol",
            "stock_name",
            "exchange",
            "industry",
            "industry_classification",
            "is_active",
        ]
    ]

    universe = universe.sort_values("baostock_code").reset_index(drop=True)

    return universe


def create_universe_table(connection):
    """Create the V2 stock-universe database table."""

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS stock_universe_v2 (
            snapshot_date TEXT NOT NULL,
            index_code TEXT NOT NULL,
            index_name TEXT NOT NULL,
            baostock_code TEXT NOT NULL,
            symbol TEXT NOT NULL,
            stock_name TEXT,
            exchange TEXT,
            industry TEXT,
            industry_classification TEXT,
            is_active INTEGER NOT NULL DEFAULT 1,
            PRIMARY KEY (
                snapshot_date,
                index_code,
                baostock_code
            )
        );
        """
    )

    connection.commit()


def save_universe_to_database(universe):
    """Save the stock-universe snapshot into SQLite."""

    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DATABASE_PATH) as connection:
        create_universe_table(connection)

        records = list(
            universe[
                [
                    "snapshot_date",
                    "index_code",
                    "index_name",
                    "baostock_code",
                    "symbol",
                    "stock_name",
                    "exchange",
                    "industry",
                    "industry_classification",
                    "is_active",
                ]
            ].itertuples(index=False, name=None)
        )

        connection.executemany(
            """
            INSERT OR REPLACE INTO stock_universe_v2 (
                snapshot_date,
                index_code,
                index_name,
                baostock_code,
                symbol,
                stock_name,
                exchange,
                industry,
                industry_classification,
                is_active
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            records,
        )

        connection.commit()


def save_universe_to_csv(universe):
    """Save a readable CSV copy for inspection."""

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    universe.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )


def main():
    print("=" * 100)
    print("V2 STOCK UNIVERSE")
    print("=" * 100)

    universe = download_hs300_universe()

    save_universe_to_database(universe)
    save_universe_to_csv(universe)

    print(f"\nSnapshot date: {universe['snapshot_date'].iloc[0]}")
    print(f"Index: {INDEX_NAME}")
    print(f"Number of stocks: {len(universe)}")
    print(f"Number of industries: {universe['industry'].nunique()}")
    print(f"Database table: stock_universe_v2")
    print(f"CSV file: {OUTPUT_FILE}")

    print("\nFirst 10 stocks:")
    print(
        universe[
            [
                "baostock_code",
                "symbol",
                "stock_name",
                "industry",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    print("\nV2 stock universe completed.")


if __name__ == "__main__":
    main()