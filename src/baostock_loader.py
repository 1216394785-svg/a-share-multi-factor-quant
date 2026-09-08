"""
Reliable V2 BaoStock historical data loader.

Features:
- automatic retry
- socket timeout
- automatic reconnection
- resumable downloads
- incremental database saving
"""

import argparse
import socket
import sqlite3
import time
from datetime import date, datetime
from pathlib import Path

import baostock as bs
import baostock.common.context as bs_context
import pandas as pd

from config.settings import DATABASE_PATH
from src.universe import result_set_to_dataframe


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = PROJECT_ROOT / "reports" / "results"
FAILURE_FILE = RESULT_DIR / "baostock_download_failures.csv"

DEFAULT_START_DATE = "2019-01-01"
DEFAULT_END_DATE = date.today().isoformat()

BAOSTOCK_FIELDS = (
    "date,code,open,high,low,close,preclose,volume,amount,"
    "adjustflag,turn,tradestatus,pctChg,peTTM,pbMRQ,"
    "psTTM,pcfNcfTTM,isST"
)

DATABASE_COLUMNS = [
    "date",
    "baostock_code",
    "symbol",
    "stock_name",
    "open",
    "high",
    "low",
    "close",
    "pre_close",
    "volume",
    "amount",
    "adjustment_flag",
    "turnover_rate",
    "trade_status",
    "pct_change",
    "pe_ttm",
    "pb_mrq",
    "ps_ttm",
    "pcf_ncf_ttm",
    "is_st",
    "source",
    "downloaded_at",
]


def create_daily_prices_table(connection):
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS daily_prices_v2 (
            date TEXT NOT NULL,
            baostock_code TEXT NOT NULL,
            symbol TEXT NOT NULL,
            stock_name TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            pre_close REAL,
            volume REAL,
            amount REAL,
            adjustment_flag INTEGER,
            turnover_rate REAL,
            trade_status INTEGER,
            pct_change REAL,
            pe_ttm REAL,
            pb_mrq REAL,
            ps_ttm REAL,
            pcf_ncf_ttm REAL,
            is_st INTEGER,
            source TEXT NOT NULL,
            downloaded_at TEXT NOT NULL,
            PRIMARY KEY (date, baostock_code)
        );
        """
    )

    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_daily_prices_v2_symbol_date
        ON daily_prices_v2 (symbol, date);
        """
    )

    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_daily_prices_v2_date
        ON daily_prices_v2 (date);
        """
    )

    connection.commit()


def load_latest_universe(connection):
    query = """
        SELECT DISTINCT
            baostock_code,
            symbol,
            stock_name
        FROM stock_universe_v2
        WHERE snapshot_date = (
            SELECT MAX(snapshot_date)
            FROM stock_universe_v2
        )
        ORDER BY baostock_code;
    """

    universe = pd.read_sql_query(query, connection)

    if universe.empty:
        raise ValueError(
            "The stock_universe_v2 table is empty. "
            "Run: python -m src.universe"
        )

    return universe


def validate_dates(start_date, end_date):
    start_timestamp = pd.Timestamp(start_date)
    end_timestamp = pd.Timestamp(end_date)

    if start_timestamp > end_timestamp:
        raise ValueError(
            "The start date must not be after the end date."
        )


def close_baostock_connection():
    """Close the BaoStock socket without waiting for a server reply."""

    current_socket = getattr(
        bs_context,
        "default_socket",
        None,
    )

    if current_socket is not None:
        try:
            current_socket.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass

        try:
            current_socket.close()
        except OSError:
            pass

    if hasattr(bs_context, "default_socket"):
        try:
            delattr(bs_context, "default_socket")
        except AttributeError:
            pass


def connect_baostock(timeout_seconds):
    """Create a fresh BaoStock connection with a timeout."""

    close_baostock_connection()

    socket.setdefaulttimeout(timeout_seconds)

    login_result = bs.login()

    if login_result.error_code != "0":
        raise ConnectionError(
            f"BaoStock login failed: "
            f"{login_result.error_code} - "
            f"{login_result.error_msg}"
        )

    current_socket = getattr(
        bs_context,
        "default_socket",
        None,
    )

    if current_socket is None:
        raise ConnectionError(
            "BaoStock did not create a valid socket."
        )

    current_socket.settimeout(timeout_seconds)


def get_completed_codes(connection, end_date):
    """
    Find stocks already downloaded near the requested end date.

    Ten calendar days are allowed for weekends, holidays
    and temporarily suspended stocks.
    """

    cutoff_date = (
        pd.Timestamp(end_date) - pd.Timedelta(days=10)
    ).strftime("%Y-%m-%d")

    query = """
        SELECT
            baostock_code,
            MAX(date) AS latest_date
        FROM daily_prices_v2
        GROUP BY baostock_code;
    """

    coverage = pd.read_sql_query(query, connection)

    if coverage.empty:
        return set()

    coverage["latest_date"] = pd.to_datetime(
        coverage["latest_date"],
        errors="coerce",
    )

    completed = coverage[
        coverage["latest_date"] >= pd.Timestamp(cutoff_date)
    ]

    return set(completed["baostock_code"])


def download_one_stock(
    baostock_code,
    symbol,
    stock_name,
    start_date,
    end_date,
):
    result = bs.query_history_k_data_plus(
        baostock_code,
        BAOSTOCK_FIELDS,
        start_date=start_date,
        end_date=end_date,
        frequency="d",
        adjustflag="2",
    )

    data = result_set_to_dataframe(result)

    if data.empty:
        raise ValueError("No historical data was returned.")

    data = data.rename(
        columns={
            "code": "baostock_code",
            "preclose": "pre_close",
            "adjustflag": "adjustment_flag",
            "turn": "turnover_rate",
            "tradestatus": "trade_status",
            "pctChg": "pct_change",
            "peTTM": "pe_ttm",
            "pbMRQ": "pb_mrq",
            "psTTM": "ps_ttm",
            "pcfNcfTTM": "pcf_ncf_ttm",
            "isST": "is_st",
        }
    )

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "pre_close",
        "volume",
        "amount",
        "adjustment_flag",
        "turnover_rate",
        "trade_status",
        "pct_change",
        "pe_ttm",
        "pb_mrq",
        "ps_ttm",
        "pcf_ncf_ttm",
        "is_st",
    ]

    for column in numeric_columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    data["symbol"] = symbol
    data["stock_name"] = stock_name
    data["source"] = "baostock"
    data["downloaded_at"] = datetime.now().isoformat(
        timespec="seconds"
    )

    data = data[DATABASE_COLUMNS]

    data = data.drop_duplicates(
        subset=["date", "baostock_code"],
        keep="last",
    )

    return data


def save_stock_data(connection, stock_data):
    clean_data = stock_data.astype(object).where(
        pd.notna(stock_data),
        None,
    )

    records = list(
        clean_data.itertuples(index=False, name=None)
    )

    placeholders = ", ".join(
        ["?"] * len(DATABASE_COLUMNS)
    )

    column_names = ", ".join(DATABASE_COLUMNS)

    connection.executemany(
        f"""
        INSERT OR REPLACE INTO daily_prices_v2 (
            {column_names}
        )
        VALUES ({placeholders});
        """,
        records,
    )

    connection.commit()

    return len(records)


def save_failures(failures):
    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    failure_frame = pd.DataFrame(
        failures,
        columns=[
            "baostock_code",
            "symbol",
            "stock_name",
            "error",
        ],
    )

    failure_frame.to_csv(
        FAILURE_FILE,
        index=False,
        encoding="utf-8-sig",
    )


def print_database_summary(connection):
    query = """
        SELECT
            COUNT(*) AS number_of_rows,
            COUNT(DISTINCT baostock_code)
                AS number_of_stocks,
            MIN(date) AS first_date,
            MAX(date) AS last_date
        FROM daily_prices_v2;
    """

    summary = pd.read_sql_query(
        query,
        connection,
    ).iloc[0]

    print("\n" + "=" * 100)
    print("V2 DATABASE SUMMARY")
    print("=" * 100)
    print(
        f"Rows stored: "
        f"{int(summary['number_of_rows']):,}"
    )
    print(
        f"Stocks stored: "
        f"{int(summary['number_of_stocks']):,}"
    )
    print(f"First date: {summary['first_date']}")
    print(f"Last date: {summary['last_date']}")


def run_download(
    start_date,
    end_date,
    limit,
    pause_seconds,
    resume,
    timeout_seconds,
    max_retries,
):
    validate_dates(start_date, end_date)

    DATABASE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with sqlite3.connect(DATABASE_PATH) as connection:
        create_daily_prices_table(connection)

        universe = load_latest_universe(connection)
        original_stock_count = len(universe)

        if resume:
            completed_codes = get_completed_codes(
                connection,
                end_date,
            )

            universe = universe[
                ~universe["baostock_code"].isin(
                    completed_codes
                )
            ].copy()

            skipped_count = (
                original_stock_count - len(universe)
            )

            print(
                f"Completed stocks skipped: "
                f"{skipped_count}"
            )

        if limit is not None:
            universe = universe.head(limit)

        print("=" * 100)
        print("RESILIENT V2 BAOSTOCK DOWNLOAD")
        print("=" * 100)
        print(f"Start date: {start_date}")
        print(f"End date: {end_date}")
        print(f"Stocks remaining: {len(universe)}")
        print(f"Maximum retries: {max_retries}")
        print(f"Socket timeout: {timeout_seconds} seconds")
        print("=" * 100)

        if universe.empty:
            print("All requested stocks are already complete.")
            print_database_summary(connection)
            return

        failures = []
        success_count = 0
        total_rows = 0
        stocks_on_current_connection = 0

        try:
            connect_baostock(timeout_seconds)

            for position, stock in enumerate(
                universe.itertuples(index=False),
                start=1,
            ):
                print(
                    f"\n[{position}/{len(universe)}] "
                    f"{stock.baostock_code} "
                    f"{stock.stock_name}"
                )

                download_succeeded = False
                final_error = ""

                for attempt in range(
                    1,
                    max_retries + 1,
                ):
                    try:
                        if stocks_on_current_connection >= 25:
                            print(
                                "    Refreshing connection..."
                            )

                            connect_baostock(
                                timeout_seconds
                            )

                            stocks_on_current_connection = 0

                        stock_data = download_one_stock(
                            baostock_code=(
                                stock.baostock_code
                            ),
                            symbol=stock.symbol,
                            stock_name=stock.stock_name,
                            start_date=start_date,
                            end_date=end_date,
                        )

                        rows_saved = save_stock_data(
                            connection,
                            stock_data,
                        )

                        success_count += 1
                        total_rows += rows_saved
                        stocks_on_current_connection += 1
                        download_succeeded = True

                        print(
                            f"    Rows saved: "
                            f"{rows_saved:,}"
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
                        stocks_on_current_connection = 0

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

                if not download_succeeded:
                    failures.append(
                        {
                            "baostock_code": (
                                stock.baostock_code
                            ),
                            "symbol": stock.symbol,
                            "stock_name": stock.stock_name,
                            "error": final_error,
                        }
                    )

                    print(
                        "    Stock skipped after "
                        "all retries."
                    )

                time.sleep(pause_seconds)

        except KeyboardInterrupt:
            print(
                "\nDownload stopped by the user. "
                "Saved data has been preserved."
            )

        finally:
            close_baostock_connection()
            save_failures(failures)
            print_database_summary(connection)

        print("\n" + "=" * 100)
        print("DOWNLOAD RESULT")
        print("=" * 100)
        print(f"Successful new stocks: {success_count}")
        print(f"Failed stocks: {len(failures)}")
        print(f"Rows processed: {total_rows:,}")
        print(f"Failure file: {FAILURE_FILE}")
        print(
            "\nYou can safely run the same "
            "--resume command again."
        )


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Reliable V2 CSI 300 historical "
            "data downloader."
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
        "--limit",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--all",
        action="store_true",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip stocks already downloaded.",
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
        default=0.10,
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    selected_limit = (
        None if args.all else args.limit
    )

    run_download(
        start_date=args.start,
        end_date=args.end,
        limit=selected_limit,
        pause_seconds=args.pause,
        resume=args.resume,
        timeout_seconds=args.timeout,
        max_retries=args.retries,
    )


if __name__ == "__main__":
    main()