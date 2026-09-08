"""
V2 market-data quality control.

Checks the daily_prices_v2 database table for:
- stock and date coverage
- duplicate observations
- missing values
- invalid OHLC relationships
- invalid prices and volumes
- trading-status and ST flags
- extreme daily returns
- missing valuation indicators
"""

import sqlite3
from pathlib import Path

import pandas as pd

from config.settings import DATABASE_PATH


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = PROJECT_ROOT / "reports" / "results"

QUALITY_FILE = RESULT_DIR / "data_quality_summary.csv"
COVERAGE_FILE = RESULT_DIR / "stock_data_coverage.csv"


def table_exists(connection, table_name):
    query = """
        SELECT COUNT(*)
        FROM sqlite_master
        WHERE type = 'table'
          AND name = ?;
    """

    result = connection.execute(
        query,
        (table_name,),
    ).fetchone()[0]

    return result > 0


def query_count(connection, query):
    result = connection.execute(query).fetchone()[0]

    if result is None:
        return 0

    return int(result)


def add_check(
    checks,
    check_name,
    issue_count,
    severity,
    description,
):
    if issue_count == 0:
        status = "PASS"
    elif severity == "CRITICAL":
        status = "FAIL"
    else:
        status = "WARNING"

    checks.append(
        {
            "check_name": check_name,
            "status": status,
            "severity": severity,
            "issue_count": issue_count,
            "description": description,
        }
    )


def load_database_summary(connection):
    query = """
        SELECT
            COUNT(*) AS number_of_rows,
            COUNT(DISTINCT baostock_code)
                AS number_of_stocks,
            MIN(date) AS first_date,
            MAX(date) AS last_date
        FROM daily_prices_v2;
    """

    return pd.read_sql_query(
        query,
        connection,
    ).iloc[0]


def load_stock_coverage(connection):
    query = """
        SELECT
            baostock_code,
            symbol,
            MAX(stock_name) AS stock_name,
            COUNT(*) AS number_of_rows,
            MIN(date) AS first_date,
            MAX(date) AS last_date,
            SUM(
                CASE
                    WHEN trade_status = 0 THEN 1
                    ELSE 0
                END
            ) AS suspended_days,
            SUM(
                CASE
                    WHEN is_st = 1 THEN 1
                    ELSE 0
                END
            ) AS st_days,
            AVG(turnover_rate) AS
                average_turnover_rate
        FROM daily_prices_v2
        GROUP BY
            baostock_code,
            symbol
        ORDER BY baostock_code;
    """

    coverage = pd.read_sql_query(
        query,
        connection,
    )

    industry_query = """
        SELECT
            baostock_code,
            industry
        FROM stock_universe_v2
        WHERE snapshot_date = (
            SELECT MAX(snapshot_date)
            FROM stock_universe_v2
        );
    """

    industries = pd.read_sql_query(
        industry_query,
        connection,
    )

    coverage = coverage.merge(
        industries,
        on="baostock_code",
        how="left",
    )

    coverage["first_date"] = pd.to_datetime(
        coverage["first_date"],
        errors="coerce",
    )

    coverage["last_date"] = pd.to_datetime(
        coverage["last_date"],
        errors="coerce",
    )

    global_last_date = coverage["last_date"].max()

    coverage["stale_days"] = (
        global_last_date - coverage["last_date"]
    ).dt.days

    coverage["coverage_status"] = "COMPLETE"

    coverage.loc[
        coverage["stale_days"] > 10,
        "coverage_status",
    ] = "STALE"

    coverage.loc[
        coverage["number_of_rows"] < 252,
        "coverage_status",
    ] = "SHORT_HISTORY"

    coverage["first_date"] = coverage[
        "first_date"
    ].dt.strftime("%Y-%m-%d")

    coverage["last_date"] = coverage[
        "last_date"
    ].dt.strftime("%Y-%m-%d")

    return coverage


def run_quality_checks(connection):
    checks = []

    summary = load_database_summary(connection)

    number_of_stocks = int(
        summary["number_of_stocks"]
    )

    universe_shortfall = max(
        0,
        250 - number_of_stocks,
    )

    add_check(
        checks,
        "Minimum stock universe size",
        universe_shortfall,
        "CRITICAL",
        "At least 250 stocks should be available.",
    )

    duplicate_count = query_count(
        connection,
        """
        SELECT COUNT(*)
        FROM (
            SELECT
                date,
                baostock_code,
                COUNT(*) AS duplicate_rows
            FROM daily_prices_v2
            GROUP BY
                date,
                baostock_code
            HAVING COUNT(*) > 1
        );
        """,
    )

    add_check(
        checks,
        "Duplicate stock-date observations",
        duplicate_count,
        "CRITICAL",
        "Each stock-date pair must be unique.",
    )

    missing_identifier_count = query_count(
        connection,
        """
        SELECT COUNT(*)
        FROM daily_prices_v2
        WHERE date IS NULL
           OR baostock_code IS NULL
           OR symbol IS NULL;
        """,
    )

    add_check(
        checks,
        "Missing identifiers",
        missing_identifier_count,
        "CRITICAL",
        "Date and stock identifiers cannot be missing.",
    )

    missing_ohlc_count = query_count(
        connection,
        """
        SELECT COUNT(*)
        FROM daily_prices_v2
        WHERE trade_status = 1
          AND (
              open IS NULL
              OR high IS NULL
              OR low IS NULL
              OR close IS NULL
          );
        """,
    )

    add_check(
        checks,
        "Missing OHLC on trading days",
        missing_ohlc_count,
        "CRITICAL",
        "Tradable observations require complete OHLC prices.",
    )

    invalid_ohlc_count = query_count(
        connection,
        """
        SELECT COUNT(*)
        FROM daily_prices_v2
        WHERE trade_status = 1
          AND (
              high < low
              OR high < open
              OR high < close
              OR low > open
              OR low > close
          );
        """,
    )

    add_check(
        checks,
        "Invalid OHLC relationships",
        invalid_ohlc_count,
        "CRITICAL",
        "High must be the highest and low the lowest price.",
    )

    nonpositive_price_count = query_count(
        connection,
        """
        SELECT COUNT(*)
        FROM daily_prices_v2
        WHERE trade_status = 1
          AND (
              open <= 0
              OR high <= 0
              OR low <= 0
              OR close <= 0
          );
        """,
    )

    add_check(
        checks,
        "Non-positive prices",
        nonpositive_price_count,
        "CRITICAL",
        "Trading-day prices must be greater than zero.",
    )

    negative_activity_count = query_count(
        connection,
        """
        SELECT COUNT(*)
        FROM daily_prices_v2
        WHERE volume < 0
           OR amount < 0
           OR turnover_rate < 0;
        """,
    )

    add_check(
        checks,
        "Negative volume or turnover",
        negative_activity_count,
        "CRITICAL",
        "Volume, amount and turnover cannot be negative.",
    )

    invalid_trade_status_count = query_count(
        connection,
        """
        SELECT COUNT(*)
        FROM daily_prices_v2
        WHERE trade_status IS NULL
           OR trade_status NOT IN (0, 1);
        """,
    )

    add_check(
        checks,
        "Invalid trading-status flags",
        invalid_trade_status_count,
        "CRITICAL",
        "Trading status must equal zero or one.",
    )

    invalid_st_count = query_count(
        connection,
        """
        SELECT COUNT(*)
        FROM daily_prices_v2
        WHERE is_st IS NULL
           OR is_st NOT IN (0, 1);
        """,
    )

    add_check(
        checks,
        "Invalid ST flags",
        invalid_st_count,
        "CRITICAL",
        "ST status must equal zero or one.",
    )

    extreme_return_count = query_count(
        connection,
        """
        SELECT COUNT(*)
        FROM daily_prices_v2
        WHERE ABS(pct_change) > 25;
        """,
    )

    add_check(
        checks,
        "Extreme daily returns above 25%",
        extreme_return_count,
        "WARNING",
        "May include IPO observations or data anomalies.",
    )

    missing_valuation_count = query_count(
        connection,
        """
        SELECT COUNT(*)
        FROM daily_prices_v2
        WHERE pe_ttm IS NULL
           OR pb_mrq IS NULL
           OR ps_ttm IS NULL
           OR pcf_ncf_ttm IS NULL;
        """,
    )

    add_check(
        checks,
        "Missing valuation indicators",
        missing_valuation_count,
        "WARNING",
        "Missing values can occur for loss-making companies.",
    )

    coverage = load_stock_coverage(connection)

    stale_stock_count = int(
        (coverage["coverage_status"] == "STALE").sum()
    )

    add_check(
        checks,
        "Stocks with stale ending dates",
        stale_stock_count,
        "WARNING",
        "The last observation is over 10 days behind.",
    )

    short_history_count = int(
        (
            coverage["coverage_status"]
            == "SHORT_HISTORY"
        ).sum()
    )

    add_check(
        checks,
        "Stocks with less than one year of data",
        short_history_count,
        "WARNING",
        "Usually caused by recent IPO listings.",
    )

    quality_summary = pd.DataFrame(checks)

    return summary, quality_summary, coverage


def main():
    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with sqlite3.connect(DATABASE_PATH) as connection:
        if not table_exists(
            connection,
            "daily_prices_v2",
        ):
            raise ValueError(
                "daily_prices_v2 does not exist. "
                "Run the BaoStock loader first."
            )

        summary, quality, coverage = (
            run_quality_checks(connection)
        )

    quality.to_csv(
        QUALITY_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    coverage.to_csv(
        COVERAGE_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print("=" * 110)
    print("V2 MARKET-DATA QUALITY REPORT")
    print("=" * 110)
    print(
        f"Rows: "
        f"{int(summary['number_of_rows']):,}"
    )
    print(
        f"Stocks: "
        f"{int(summary['number_of_stocks']):,}"
    )
    print(f"First date: {summary['first_date']}")
    print(f"Last date: {summary['last_date']}")

    print("\n" + "=" * 110)
    print("QUALITY CHECKS")
    print("=" * 110)

    print(
        quality[
            [
                "check_name",
                "status",
                "issue_count",
                "description",
            ]
        ].to_string(index=False)
    )

    failure_count = int(
        (quality["status"] == "FAIL").sum()
    )

    warning_count = int(
        (quality["status"] == "WARNING").sum()
    )

    print("\n" + "=" * 110)

    if failure_count == 0:
        print("OVERALL RESULT: PASS")
    else:
        print("OVERALL RESULT: FAIL")

    print(f"Critical failures: {failure_count}")
    print(f"Warnings: {warning_count}")
    print(f"Quality report: {QUALITY_FILE}")
    print(f"Coverage report: {COVERAGE_FILE}")

    if failure_count > 0:
        raise SystemExit(
            "Critical data-quality problems were detected."
        )

    print("\nV2 data-quality analysis completed.")


if __name__ == "__main__":
    main()