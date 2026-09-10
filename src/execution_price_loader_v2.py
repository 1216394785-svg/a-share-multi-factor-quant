"""
Download unadjusted execution prices for the V2 event-driven backtest.

Why this table is separate:

- Adjusted prices are appropriate for factor calculations.
- Unadjusted prices are required for realistic share quantities,
  board-lot calculations, commissions and cash accounting.
- Only stocks that appear in portfolio_targets_v2 are downloaded.
- CSI 300 index prices are also downloaded as the benchmark.

The command can safely be rerun with --resume.
"""

from __future__ import annotations

import argparse
import socket
import sqlite3
import time
from datetime import timedelta
from pathlib import Path

import baostock as bs
import numpy as np
import pandas as pd

from src.factors_v2 import resolve_database_path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "reports" / "results"

TARGET_TABLE = "portfolio_targets_v2"
FACTOR_PRICE_TABLE = "daily_prices_v2"
OUTPUT_TABLE = "execution_prices_v2"

COVERAGE_FILE = (
    RESULTS_DIR / "execution_price_coverage_v2.csv"
)
FAILURE_FILE = (
    RESULTS_DIR / "execution_price_failures_v2.csv"
)

BENCHMARK_CODE = "sh.000300"
BENCHMARK_NAME = "CSI 300"

SOCKET_TIMEOUT_SECONDS = 45
MAXIMUM_RETRIES = 4
RECONNECT_INTERVAL = 25

STOCK_FIELDS = (
    "date,code,open,high,low,close,preclose,"
    "volume,amount,tradestatus,pctChg,isST"
)

INDEX_FIELDS = (
    "date,code,open,high,low,close,preclose,"
    "volume,amount,tradestatus,pctChg"
)


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


def initialise_execution_price_table(
    connection: sqlite3.Connection,
) -> None:
    connection.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {OUTPUT_TABLE} (
            date TEXT NOT NULL,
            baostock_code TEXT NOT NULL,
            symbol TEXT,
            stock_name TEXT,
            instrument_type TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            pre_close REAL,
            volume REAL,
            amount REAL,
            trade_status INTEGER,
            pct_change REAL,
            is_st INTEGER,
            PRIMARY KEY (
                date,
                baostock_code
            )
        );
        """
    )

    connection.execute(
        f"""
        CREATE INDEX IF NOT EXISTS
        idx_{OUTPUT_TABLE}_code_date
        ON {OUTPUT_TABLE} (
            baostock_code,
            date
        );
        """
    )

    connection.commit()


def load_target_instruments(
    connection: sqlite3.Connection,
) -> pd.DataFrame:
    targets = pd.read_sql_query(
        f"""
        SELECT DISTINCT
            baostock_code,
            symbol,
            stock_name
        FROM {TARGET_TABLE}
        ORDER BY baostock_code;
        """,
        connection,
    )

    targets["instrument_type"] = "STOCK"

    benchmark = pd.DataFrame(
        [
            {
                "baostock_code": BENCHMARK_CODE,
                "symbol": "000300.SH",
                "stock_name": BENCHMARK_NAME,
                "instrument_type": "INDEX",
            }
        ]
    )

    instruments = pd.concat(
        [targets, benchmark],
        ignore_index=True,
    )

    instruments = instruments.drop_duplicates(
        "baostock_code",
        keep="last",
    )

    return instruments


def determine_default_dates(
    connection: sqlite3.Connection,
) -> tuple[str, str]:
    target_dates = pd.read_sql_query(
        f"""
        SELECT
            MIN(signal_price_date) AS first_date
        FROM {TARGET_TABLE};
        """,
        connection,
    )

    price_dates = pd.read_sql_query(
        f"""
        SELECT
            MAX(date) AS last_date
        FROM {FACTOR_PRICE_TABLE};
        """,
        connection,
    )

    first_date = pd.to_datetime(
        target_dates["first_date"].iloc[0],
        errors="coerce",
    )

    last_date = pd.to_datetime(
        price_dates["last_date"].iloc[0],
        errors="coerce",
    )

    if pd.isna(first_date) or pd.isna(last_date):
        raise ValueError(
            "Unable to determine the execution-price date range."
        )

    start_date = (
        first_date - timedelta(days=15)
    ).date().isoformat()

    end_date = last_date.date().isoformat()

    return start_date, end_date


def login_baostock() -> None:
    socket.setdefaulttimeout(
        SOCKET_TIMEOUT_SECONDS
    )

    login_result = bs.login()

    if login_result.error_code != "0":
        raise ConnectionError(
            "BaoStock login failed: "
            f"{login_result.error_code} "
            f"{login_result.error_msg}"
        )


def reconnect_baostock() -> None:
    try:
        bs.logout()
    except Exception:
        pass

    time.sleep(1.0)
    login_baostock()


def get_resume_start_date(
    connection: sqlite3.Connection,
    baostock_code: str,
    requested_start_date: str,
    requested_end_date: str,
    resume: bool,
) -> str | None:
    if not resume:
        return requested_start_date

    result = connection.execute(
        f"""
        SELECT MAX(date)
        FROM {OUTPUT_TABLE}
        WHERE baostock_code = ?;
        """,
        (baostock_code,),
    ).fetchone()

    latest_saved_date = (
        result[0]
        if result and result[0]
        else None
    )

    if latest_saved_date is None:
        return requested_start_date

    next_date = (
        pd.Timestamp(latest_saved_date)
        + pd.Timedelta(days=1)
    ).date().isoformat()

    actual_start_date = max(
        requested_start_date,
        next_date,
    )

    if actual_start_date > requested_end_date:
        return None

    return actual_start_date


def download_one_instrument(
    baostock_code: str,
    stock_name: str,
    symbol: str,
    instrument_type: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    fields = (
        INDEX_FIELDS
        if instrument_type == "INDEX"
        else STOCK_FIELDS
    )

    result = bs.query_history_k_data_plus(
        baostock_code,
        fields,
        start_date=start_date,
        end_date=end_date,
        frequency="d",
        adjustflag="3",
    )

    if result.error_code != "0":
        raise RuntimeError(
            f"{result.error_code}: "
            f"{result.error_msg}"
        )

    rows = []

    while (
        result.error_code == "0"
        and result.next()
    ):
        rows.append(result.get_row_data())

    if result.error_code != "0":
        raise RuntimeError(
            f"{result.error_code}: "
            f"{result.error_msg}"
        )

    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame(
        rows,
        columns=result.fields,
    )

    rename_map = {
        "code": "baostock_code",
        "preclose": "pre_close",
        "tradestatus": "trade_status",
        "pctChg": "pct_change",
        "isST": "is_st",
    }

    frame = frame.rename(
        columns=rename_map
    )

    if "is_st" not in frame.columns:
        frame["is_st"] = 0

    frame["symbol"] = symbol
    frame["stock_name"] = stock_name
    frame["instrument_type"] = instrument_type

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "pre_close",
        "volume",
        "amount",
        "trade_status",
        "pct_change",
        "is_st",
    ]

    for column in numeric_columns:
        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

    frame["date"] = pd.to_datetime(
        frame["date"],
        errors="coerce",
    ).dt.strftime("%Y-%m-%d")

    frame = frame.dropna(
        subset=[
            "date",
            "baostock_code",
        ]
    )

    output_columns = [
        "date",
        "baostock_code",
        "symbol",
        "stock_name",
        "instrument_type",
        "open",
        "high",
        "low",
        "close",
        "pre_close",
        "volume",
        "amount",
        "trade_status",
        "pct_change",
        "is_st",
    ]

    return frame[output_columns]


def convert_database_value(value):
    if pd.isna(value):
        return None

    if isinstance(value, np.generic):
        return value.item()

    return value


def save_instrument_prices(
    connection: sqlite3.Connection,
    frame: pd.DataFrame,
) -> int:
    if frame.empty:
        return 0

    columns = [
        "date",
        "baostock_code",
        "symbol",
        "stock_name",
        "instrument_type",
        "open",
        "high",
        "low",
        "close",
        "pre_close",
        "volume",
        "amount",
        "trade_status",
        "pct_change",
        "is_st",
    ]

    placeholders = ", ".join(
        ["?"] * len(columns)
    )

    column_text = ", ".join(columns)

    sql = f"""
        INSERT OR REPLACE INTO {OUTPUT_TABLE}
        ({column_text})
        VALUES ({placeholders});
    """

    values = [
        tuple(
            convert_database_value(
                row[column]
            )
            for column in columns
        )
        for _, row in frame.iterrows()
    ]

    connection.executemany(
        sql,
        values,
    )

    connection.commit()

    return len(values)


def download_with_retries(
    instrument: pd.Series,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    last_error: Exception | None = None

    for attempt in range(
        1,
        MAXIMUM_RETRIES + 1,
    ):
        try:
            return download_one_instrument(
                baostock_code=instrument[
                    "baostock_code"
                ],
                stock_name=instrument[
                    "stock_name"
                ],
                symbol=instrument["symbol"],
                instrument_type=instrument[
                    "instrument_type"
                ],
                start_date=start_date,
                end_date=end_date,
            )

        except (
            OSError,
            ConnectionError,
            RuntimeError,
            socket.timeout,
        ) as error:
            last_error = error

            print(
                f"    Attempt {attempt}/"
                f"{MAXIMUM_RETRIES} failed: "
                f"{error}",
                flush=True,
            )

            if attempt < MAXIMUM_RETRIES:
                try:
                    reconnect_baostock()
                except Exception as reconnect_error:
                    print(
                        "    Reconnect failed: "
                        f"{reconnect_error}",
                        flush=True,
                    )

                time.sleep(
                    min(2.0 * attempt, 8.0)
                )

    raise RuntimeError(
        f"Download failed after "
        f"{MAXIMUM_RETRIES} attempts: "
        f"{last_error}"
    )


def build_coverage_report(
    connection: sqlite3.Connection,
    instruments: pd.DataFrame,
) -> pd.DataFrame:
    coverage = pd.read_sql_query(
        f"""
        SELECT
            baostock_code,
            MIN(date) AS first_date,
            MAX(date) AS last_date,
            COUNT(*) AS number_of_rows,
            SUM(
                CASE
                    WHEN trade_status = 1
                    THEN 1
                    ELSE 0
                END
            ) AS trading_rows
        FROM {OUTPUT_TABLE}
        GROUP BY baostock_code;
        """,
        connection,
    )

    coverage = instruments.merge(
        coverage,
        on="baostock_code",
        how="left",
        validate="one_to_one",
    )

    coverage[
        "number_of_rows"
    ] = coverage[
        "number_of_rows"
    ].fillna(0).astype(int)

    coverage[
        "trading_rows"
    ] = coverage[
        "trading_rows"
    ].fillna(0).astype(int)

    coverage["coverage_status"] = np.where(
        coverage["number_of_rows"] > 0,
        "AVAILABLE",
        "MISSING",
    )

    return coverage.sort_values(
        [
            "coverage_status",
            "baostock_code",
        ]
    ).reset_index(drop=True)


def run_download(
    start_date: str | None,
    end_date: str | None,
    resume: bool,
    pause_seconds: float,
) -> None:
    database_path = resolve_database_path()
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 120)
    print("V2 UNADJUSTED EXECUTION-PRICE DOWNLOAD")
    print("=" * 120)
    print(f"Database: {database_path}")

    with sqlite3.connect(
        database_path
    ) as connection:
        for required_table in [
            TARGET_TABLE,
            FACTOR_PRICE_TABLE,
        ]:
            if not table_exists(
                connection,
                required_table,
            ):
                raise RuntimeError(
                    "Required table does not exist: "
                    f"{required_table}"
                )

        initialise_execution_price_table(
            connection
        )

        default_start, default_end = (
            determine_default_dates(
                connection
            )
        )

        actual_start = (
            start_date or default_start
        )
        actual_end = end_date or default_end

        if actual_start > actual_end:
            raise ValueError(
                "Start date must not be after end date."
            )

        instruments = load_target_instruments(
            connection
        )

        print(f"Start date: {actual_start}")
        print(f"End date: {actual_end}")
        print(
            f"Instruments: {len(instruments):,}"
        )
        print(
            f"Stocks: "
            f"{(instruments['instrument_type'] == 'STOCK').sum():,}"
        )
        print("Benchmark indexes: 1")
        print(f"Resume mode: {resume}")
        print()

        login_baostock()

        failures = []
        processed_rows = 0
        successful_instruments = 0
        skipped_instruments = 0

        try:
            total = len(instruments)

            for position, (
                _,
                instrument,
            ) in enumerate(
                instruments.iterrows(),
                start=1,
            ):
                if (
                    position > 1
                    and (
                        position - 1
                    ) % RECONNECT_INTERVAL
                    == 0
                ):
                    print(
                        "Reconnecting BaoStock...",
                        flush=True,
                    )
                    reconnect_baostock()

                code = instrument[
                    "baostock_code"
                ]
                name = instrument[
                    "stock_name"
                ]

                instrument_start = (
                    get_resume_start_date(
                        connection=connection,
                        baostock_code=code,
                        requested_start_date=(
                            actual_start
                        ),
                        requested_end_date=(
                            actual_end
                        ),
                        resume=resume,
                    )
                )

                print(
                    f"[{position}/{total}] "
                    f"{code} {name}",
                    flush=True,
                )

                if instrument_start is None:
                    print(
                        "    Already complete.",
                        flush=True,
                    )
                    skipped_instruments += 1
                    continue

                try:
                    data = download_with_retries(
                        instrument=instrument,
                        start_date=(
                            instrument_start
                        ),
                        end_date=actual_end,
                    )

                    saved_rows = (
                        save_instrument_prices(
                            connection,
                            data,
                        )
                    )

                    processed_rows += saved_rows
                    successful_instruments += 1

                    print(
                        f"    Rows saved: "
                        f"{saved_rows:,}",
                        flush=True,
                    )

                except Exception as error:
                    failures.append(
                        {
                            "baostock_code": code,
                            "symbol": instrument[
                                "symbol"
                            ],
                            "stock_name": name,
                            "instrument_type": (
                                instrument[
                                    "instrument_type"
                                ]
                            ),
                            "start_date": (
                                instrument_start
                            ),
                            "end_date": actual_end,
                            "error": str(error),
                        }
                    )

                    print(
                        f"    FAILED: {error}",
                        flush=True,
                    )

                time.sleep(
                    max(0.0, pause_seconds)
                )

        finally:
            try:
                bs.logout()
            except Exception:
                pass

        coverage = build_coverage_report(
            connection,
            instruments,
        )

    coverage.to_csv(
        COVERAGE_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    failure_frame = pd.DataFrame(
        failures
    )

    failure_frame.to_csv(
        FAILURE_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    missing_count = int(
        (
            coverage["coverage_status"]
            == "MISSING"
        ).sum()
    )

    total_database_rows = int(
        coverage["number_of_rows"].sum()
    )

    print()
    print("=" * 120)
    print("EXECUTION-PRICE DOWNLOAD RESULT")
    print("=" * 120)
    print(
        f"Successful instruments this run: "
        f"{successful_instruments:,}"
    )
    print(
        f"Already complete: "
        f"{skipped_instruments:,}"
    )
    print(
        f"Rows processed this run: "
        f"{processed_rows:,}"
    )
    print(
        f"Total database rows: "
        f"{total_database_rows:,}"
    )
    print(
        f"Missing instruments: "
        f"{missing_count:,}"
    )
    print(
        f"Failures this run: "
        f"{len(failures):,}"
    )
    print(f"Coverage file: {COVERAGE_FILE}")
    print(f"Failure file: {FAILURE_FILE}")

    if missing_count:
        print()
        print(
            "Some instruments are still missing. "
            "Run the same --resume command again."
        )
    else:
        print()
        print(
            "All execution-price instruments "
            "are available."
        )

    print()
    print(
        "V2 execution-price download completed."
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download unadjusted execution prices "
            "for the V2 backtest."
        )
    )

    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Optional start date: YYYY-MM-DD",
    )

    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="Optional end date: YYYY-MM-DD",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume after the latest saved date.",
    )

    parser.add_argument(
        "--pause",
        type=float,
        default=0.05,
        help="Pause between instruments.",
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    run_download(
        start_date=arguments.start,
        end_date=arguments.end,
        resume=arguments.resume,
        pause_seconds=arguments.pause,
    )


if __name__ == "__main__":
    main()