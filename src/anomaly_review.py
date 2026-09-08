"""
Review unusual observations found by the V2 data-quality module.
"""

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import DATABASE_PATH


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = PROJECT_ROOT / "reports" / "results"

EXTREME_FILE = RESULT_DIR / "extreme_return_review.csv"
SHORT_HISTORY_FILE = RESULT_DIR / "short_history_review.csv"


def load_extreme_returns(connection):
    query = """
        WITH ranked_observations AS (
            SELECT
                date,
                baostock_code,
                symbol,
                stock_name,
                open,
                high,
                low,
                close,
                pre_close,
                pct_change,
                trade_status,
                is_st,
                ROW_NUMBER() OVER (
                    PARTITION BY baostock_code
                    ORDER BY date
                ) AS observation_number
            FROM daily_prices_v2
            WHERE trade_status = 1
        )
        SELECT *
        FROM ranked_observations
        WHERE ABS(pct_change) > 25
        ORDER BY ABS(pct_change) DESC;
    """

    extreme = pd.read_sql_query(
        query,
        connection,
    )

    if extreme.empty:
        return extreme

    extreme["calculated_return_percent"] = np.where(
        extreme["pre_close"] > 0,
        (
            extreme["close"]
            / extreme["pre_close"]
            - 1
        )
        * 100,
        np.nan,
    )

    extreme["return_difference"] = (
        extreme["pct_change"]
        - extreme["calculated_return_percent"]
    ).abs()

    conditions = [
        extreme["observation_number"] <= 5,
        extreme["return_difference"] > 0.50,
        extreme["is_st"] == 1,
    ]

    classifications = [
        "LIKELY_NEW_LISTING",
        "PRICE_RETURN_MISMATCH",
        "ST_OBSERVATION_REVIEW",
    ]

    extreme["review_classification"] = np.select(
        conditions,
        classifications,
        default="UNUSUAL_MOVE_REVIEW",
    )

    return extreme


def load_short_history_stocks(connection):
    query = """
        SELECT
            baostock_code,
            symbol,
            MAX(stock_name) AS stock_name,
            COUNT(*) AS number_of_rows,
            MIN(date) AS first_date,
            MAX(date) AS last_date
        FROM daily_prices_v2
        GROUP BY
            baostock_code,
            symbol
        HAVING COUNT(*) < 252
        ORDER BY number_of_rows;
    """

    return pd.read_sql_query(
        query,
        connection,
    )


def main():
    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with sqlite3.connect(DATABASE_PATH) as connection:
        extreme = load_extreme_returns(connection)
        short_history = load_short_history_stocks(
            connection
        )

    extreme.to_csv(
        EXTREME_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    short_history.to_csv(
        SHORT_HISTORY_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print("=" * 110)
    print("V2 DATA ANOMALY REVIEW")
    print("=" * 110)

    print(
        f"Extreme-return observations: "
        f"{len(extreme)}"
    )

    if not extreme.empty:
        classification_summary = (
            extreme["review_classification"]
            .value_counts()
            .rename_axis("classification")
            .reset_index(name="number_of_observations")
        )

        print("\nClassification summary:")
        print(
            classification_summary.to_string(
                index=False
            )
        )

        print("\nLargest 20 absolute returns:")
        print(
            extreme[
                [
                    "date",
                    "baostock_code",
                    "stock_name",
                    "pct_change",
                    "calculated_return_percent",
                    "observation_number",
                    "review_classification",
                ]
            ]
            .head(20)
            .to_string(index=False)
        )

    print("\n" + "=" * 110)
    print("SHORT-HISTORY STOCKS")
    print("=" * 110)

    if short_history.empty:
        print("No stocks have less than 252 observations.")
    else:
        print(
            short_history.to_string(index=False)
        )

    print("\n" + "=" * 110)
    print(f"Extreme-return report: {EXTREME_FILE}")
    print(f"Short-history report: {SHORT_HISTORY_FILE}")
    print("Anomaly review completed.")


if __name__ == "__main__":
    main()