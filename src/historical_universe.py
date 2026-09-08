"""
Historical CSI 300 constituent snapshots.

Downloads monthly point-in-time CSI 300 membership to reduce
survivorship bias in historical factor research and backtesting.
"""

import argparse
import sqlite3
import time
from datetime import date
from pathlib import Path

import baostock as bs
import pandas as pd

from config.settings import DATABASE_PATH
from src.baostock_loader import (
    close_baostock_connection,
    connect_baostock,
)
from src.universe import (
    baostock_to_yahoo_symbol,
    result_set_to_dataframe,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = PROJECT_ROOT / "reports" / "results"

HISTORY_FILE = (
    RESULT_DIR / "hs300_constituent_history.csv"
)
SUMMARY_FILE = (
    RESULT_DIR / "historical_universe_summary.csv"
)
FAILURE_FILE = (
    RESULT_DIR / "historical_universe_failures.csv"
)

INDEX_CODE = "000300"
INDEX_NAME = "CSI 300"

DEFAULT_START_DATE = "2019-01-01"
DEFAULT_END_DATE = date.today().isoformat()


def create_history_table(connection):
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS
        index_constituents_history (
            query_date TEXT NOT NULL,
            update_date TEXT,
            index_code TEXT NOT NULL,
            index_name TEXT NOT NULL,
            baostock_code TEXT NOT NULL,
            symbol TEXT NOT NULL,
            stock_name TEXT,
            source TEXT NOT NULL,
            PRIMARY KEY (
                query_date,
                index_code,
                baostock_code
            )
        );
        """
    )

    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS
        idx_constituents_history_code_date
        ON index_constituents_history (
            baostock_code,
            query_date
        );
        """
    )

    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS
        idx_constituents_history_query_date
        ON index_constituents_history (
            query_date
        );
        """
    )

    connection.commit()


def generate_snapshot_dates(start_date, end_date):
    start_timestamp = pd.Timestamp(start_date)
    end_timestamp = pd.Timestamp(end_date)

    if start_timestamp > end_timestamp:
        raise ValueError(
            "Start date must not be after end date."
        )

    month_ends = pd.date_range(
        start=start_timestamp,
        end=end_timestamp,
        freq="ME",
    )

    snapshot_dates = {
        timestamp.strftime("%Y-%m-%d")
        for timestamp in month_ends
    }

    snapshot_dates.add(
        start_timestamp.strftime("%Y-%m-%d")
    )

    snapshot_dates.add(
        end_timestamp.strftime("%Y-%m-%d")
    )

    return sorted(snapshot_dates)


def get_completed_snapshot_dates(connection):
    query = """
        SELECT query_date
        FROM index_constituents_history
        WHERE index_code = ?
        GROUP BY query_date
        HAVING COUNT(DISTINCT baostock_code) >= 250;
    """

    rows = connection.execute(
        query,
        (INDEX_CODE,),
    ).fetchall()

    return {row[0] for row in rows}


def download_snapshot(query_date):
    result = bs.query_hs300_stocks(query_date)

    snapshot = result_set_to_dataframe(result)

    if snapshot.empty:
        raise ValueError(
            f"No constituent data returned for "
            f"{query_date}."
        )

    if len(snapshot) < 250:
        raise ValueError(
            f"Only {len(snapshot)} constituents returned "
            f"for {query_date}."
        )

    if "updateDate" not in snapshot.columns:
        snapshot["updateDate"] = query_date

    snapshot = snapshot.rename(
        columns={
            "updateDate": "update_date",
            "code": "baostock_code",
            "code_name": "stock_name",
        }
    )

    snapshot["symbol"] = snapshot[
        "baostock_code"
    ].apply(baostock_to_yahoo_symbol)

    snapshot["query_date"] = query_date
    snapshot["index_code"] = INDEX_CODE
    snapshot["index_name"] = INDEX_NAME
    snapshot["source"] = "baostock"

    snapshot = snapshot[
        [
            "query_date",
            "update_date",
            "index_code",
            "index_name",
            "baostock_code",
            "symbol",
            "stock_name",
            "source",
        ]
    ]

    snapshot = snapshot.drop_duplicates(
        subset=["baostock_code"],
        keep="last",
    )

    return snapshot


def save_snapshot(connection, snapshot):
    query_date = snapshot["query_date"].iloc[0]

    connection.execute(
        """
        DELETE FROM index_constituents_history
        WHERE query_date = ?
          AND index_code = ?;
        """,
        (query_date, INDEX_CODE),
    )

    clean_snapshot = snapshot.astype(object).where(
        pd.notna(snapshot),
        None,
    )

    records = list(
        clean_snapshot.itertuples(
            index=False,
            name=None,
        )
    )

    connection.executemany(
        """
        INSERT OR REPLACE INTO
        index_constituents_history (
            query_date,
            update_date,
            index_code,
            index_name,
            baostock_code,
            symbol,
            stock_name,
            source
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?);
        """,
        records,
    )

    connection.commit()

    return len(records)


def export_results(connection):
    history_query = """
        SELECT *
        FROM index_constituents_history
        WHERE index_code = ?
        ORDER BY
            query_date,
            baostock_code;
    """

    history = pd.read_sql_query(
        history_query,
        connection,
        params=(INDEX_CODE,),
    )

    if history.empty:
        raise ValueError(
            "Historical constituent table is empty."
        )

    latest_query_date = history[
        "query_date"
    ].max()

    current_codes = set(
        history.loc[
            history["query_date"]
            == latest_query_date,
            "baostock_code",
        ]
    )

    historical_codes = set(
        history["baostock_code"]
    )

    summary = pd.DataFrame(
        [
            {
                "index_code": INDEX_CODE,
                "index_name": INDEX_NAME,
                "first_snapshot_date": (
                    history["query_date"].min()
                ),
                "last_snapshot_date": (
                    latest_query_date
                ),
                "number_of_snapshots": (
                    history["query_date"].nunique()
                ),
                "current_constituents": (
                    len(current_codes)
                ),
                "unique_historical_constituents": (
                    len(historical_codes)
                ),
                "former_constituents": (
                    len(
                        historical_codes
                        - current_codes
                    )
                ),
            }
        ]
    )

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    history.to_csv(
        HISTORY_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    return history, summary


def save_failures(failures):
    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    failure_frame = pd.DataFrame(
        failures,
        columns=[
            "query_date",
            "error",
        ],
    )

    failure_frame.to_csv(
        FAILURE_FILE,
        index=False,
        encoding="utf-8-sig",
    )


def run_download(
    start_date,
    end_date,
    resume,
    timeout_seconds,
    max_retries,
    pause_seconds,
):
    snapshot_dates = generate_snapshot_dates(
        start_date,
        end_date,
    )

    DATABASE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with sqlite3.connect(DATABASE_PATH) as connection:
        create_history_table(connection)

        if resume:
            completed_dates = (
                get_completed_snapshot_dates(
                    connection
                )
            )

            original_count = len(snapshot_dates)

            snapshot_dates = [
                snapshot_date
                for snapshot_date in snapshot_dates
                if snapshot_date
                not in completed_dates
            ]

            print(
                f"Completed snapshots skipped: "
                f"{original_count - len(snapshot_dates)}"
            )

        print("=" * 100)
        print("HISTORICAL CSI 300 CONSTITUENTS")
        print("=" * 100)
        print(f"Start date: {start_date}")
        print(f"End date: {end_date}")
        print(
            f"Snapshots remaining: "
            f"{len(snapshot_dates)}"
        )
        print("=" * 100)

        failures = []
        successful_snapshots = 0
        connection_usage = 0

        try:
            if snapshot_dates:
                connect_baostock(timeout_seconds)

            for position, snapshot_date in enumerate(
                snapshot_dates,
                start=1,
            ):
                print(
                    f"\n[{position}/{len(snapshot_dates)}] "
                    f"{snapshot_date}"
                )

                success = False
                final_error = ""

                for attempt in range(
                    1,
                    max_retries + 1,
                ):
                    try:
                        if connection_usage >= 20:
                            print(
                                "    Refreshing connection..."
                            )

                            connect_baostock(
                                timeout_seconds
                            )

                            connection_usage = 0

                        snapshot = download_snapshot(
                            snapshot_date
                        )

                        rows_saved = save_snapshot(
                            connection,
                            snapshot,
                        )

                        successful_snapshots += 1
                        connection_usage += 1
                        success = True

                        update_date = snapshot[
                            "update_date"
                        ].iloc[0]

                        print(
                            f"    Rows saved: {rows_saved}"
                        )
                        print(
                            f"    Source update date: "
                            f"{update_date}"
                        )

                        break

                    except KeyboardInterrupt:
                        raise

                    except Exception as error:
                        final_error = str(error)

                        print(
                            f"    Attempt "
                            f"{attempt}/{max_retries} "
                            f"failed: {error}"
                        )

                        close_baostock_connection()
                        connection_usage = 0

                        if attempt < max_retries:
                            wait_seconds = 5 * attempt

                            print(
                                f"    Reconnecting in "
                                f"{wait_seconds} seconds..."
                            )

                            time.sleep(wait_seconds)

                            try:
                                connect_baostock(
                                    timeout_seconds
                                )
                            except Exception as login_error:
                                final_error = str(
                                    login_error
                                )

                if not success:
                    failures.append(
                        {
                            "query_date": snapshot_date,
                            "error": final_error,
                        }
                    )

                time.sleep(pause_seconds)

        except KeyboardInterrupt:
            print(
                "\nDownload stopped by the user. "
                "Completed snapshots were preserved."
            )

        finally:
            close_baostock_connection()
            save_failures(failures)

        history, summary = export_results(
            connection
        )

    result = summary.iloc[0]

    print("\n" + "=" * 100)
    print("HISTORICAL UNIVERSE SUMMARY")
    print("=" * 100)
    print(
        f"Snapshots stored: "
        f"{result['number_of_snapshots']}"
    )
    print(
        f"Current constituents: "
        f"{result['current_constituents']}"
    )
    print(
        f"Unique historical constituents: "
        f"{result['unique_historical_constituents']}"
    )
    print(
        f"Former constituents: "
        f"{result['former_constituents']}"
    )
    print(
        f"Failed snapshots this run: "
        f"{len(failures)}"
    )
    print(f"History CSV: {HISTORY_FILE}")
    print(f"Summary CSV: {SUMMARY_FILE}")
    print("\nHistorical universe completed.")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Download monthly historical "
            "CSI 300 constituent snapshots."
        )
    )

    parser.add_argument(
        "--start",
        default=DEFAULT_START_DATE,
    )

    parser.add_argument(
        "--end",
        default=DEFAULT_END_DATE,
    )

    parser.add_argument(
        "--resume",
        action="store_true",
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=45,
    )

    parser.add_argument(
        "--retries",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--pause",
        type=float,
        default=0.20,
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    run_download(
        start_date=args.start,
        end_date=args.end,
        resume=args.resume,
        timeout_seconds=args.timeout,
        max_retries=args.retries,
        pause_seconds=args.pause,
    )


if __name__ == "__main__":
    main()