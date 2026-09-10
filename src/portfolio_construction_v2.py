"""
V2 portfolio construction.

Portfolio rules:

- Hold up to 20 stocks
- Equal target weight of 5% per stock
- Maximum 3 stocks from the same industry
- New stocks normally need to rank in the top 30
- Existing holdings are retained while rank is 40 or better
- Monthly rebalancing
- Cash is allowed when fewer than 20 stocks satisfy constraints
- Generate auditable target holdings and rebalance orders
"""

from __future__ import annotations

import sqlite3
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from src.factors_v2 import resolve_database_path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "reports" / "results"

SCORE_TABLE = "walk_forward_stock_scores_v2"

TARGET_TABLE = "portfolio_targets_v2"
ORDER_TABLE = "portfolio_rebalance_orders_v2"
SUMMARY_TABLE = "portfolio_construction_summary_v2"

TARGET_FILE = (
    RESULTS_DIR / "portfolio_targets_v2.csv"
)
ORDER_FILE = (
    RESULTS_DIR / "portfolio_rebalance_orders_v2.csv"
)
SUMMARY_FILE = (
    RESULTS_DIR / "portfolio_construction_summary_v2.csv"
)

TARGET_NUMBER_OF_HOLDINGS = 20
TARGET_STOCK_WEIGHT = 1.0 / TARGET_NUMBER_OF_HOLDINGS

MAXIMUM_STOCKS_PER_INDUSTRY = 3

ENTRY_RANK_LIMIT = 30
EXIT_RANK_LIMIT = 40

WEIGHT_TOLERANCE = 1e-8


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


def load_walk_forward_scores(
    connection: sqlite3.Connection,
) -> pd.DataFrame:
    scores = pd.read_sql_query(
        f'SELECT * FROM "{SCORE_TABLE}"',
        connection,
    )

    required_columns = {
        "snapshot_date",
        "signal_price_date",
        "baostock_code",
        "industry",
        "walk_forward_score",
        "walk_forward_rank",
        "model_ready",
    }

    missing_columns = required_columns.difference(
        scores.columns
    )

    if missing_columns:
        raise ValueError(
            f"{SCORE_TABLE} is missing columns: "
            f"{sorted(missing_columns)}"
        )

    scores["snapshot_date"] = pd.to_datetime(
        scores["snapshot_date"],
        errors="coerce",
    )

    scores["signal_price_date"] = pd.to_datetime(
        scores["signal_price_date"],
        errors="coerce",
    )

    numeric_columns = [
        "close",
        "walk_forward_score",
        "walk_forward_rank",
        "model_ready",
    ]

    for column in numeric_columns:
        if column in scores.columns:
            scores[column] = pd.to_numeric(
                scores[column],
                errors="coerce",
            )

    scores["industry"] = (
        scores["industry"]
        .fillna("UNKNOWN")
        .astype(str)
        .str.strip()
        .replace("", "UNKNOWN")
    )

    scores = scores.loc[
        scores["model_ready"] == 1
    ].copy()

    scores = scores.dropna(
        subset=[
            "snapshot_date",
            "baostock_code",
            "walk_forward_score",
            "walk_forward_rank",
        ]
    )

    duplicate_count = int(
        scores.duplicated(
            [
                "snapshot_date",
                "baostock_code",
            ]
        ).sum()
    )

    if duplicate_count:
        raise ValueError(
            f"{SCORE_TABLE} contains "
            f"{duplicate_count:,} duplicate rows."
        )

    return scores.sort_values(
        [
            "snapshot_date",
            "walk_forward_rank",
        ]
    ).reset_index(drop=True)


def can_add_industry(
    industry: str,
    industry_counts: Counter,
) -> bool:
    return (
        industry_counts[industry]
        < MAXIMUM_STOCKS_PER_INDUSTRY
    )


def create_selected_record(
    row: pd.Series,
    selection_status: str,
    selection_reason: str,
) -> dict:
    record = row.to_dict()

    record["selection_status"] = (
        selection_status
    )
    record["selection_reason"] = (
        selection_reason
    )
    record["target_weight"] = (
        TARGET_STOCK_WEIGHT
    )

    return record


def select_monthly_portfolio(
    candidates: pd.DataFrame,
    previous_holdings: dict[str, dict],
) -> pd.DataFrame:
    candidates = candidates.sort_values(
        "walk_forward_rank"
    ).copy()

    selected_records: list[dict] = []
    selected_codes: set[str] = set()
    industry_counts: Counter = Counter()

    previous_codes = set(
        previous_holdings.keys()
    )

    # Step 1: retain existing holdings that remain
    # inside the exit buffer.
    retention_pool = candidates.loc[
        candidates["baostock_code"].isin(
            previous_codes
        )
        & (
            candidates["walk_forward_rank"]
            <= EXIT_RANK_LIMIT
        )
    ].sort_values(
        "walk_forward_rank"
    )

    for _, row in retention_pool.iterrows():
        if (
            len(selected_records)
            >= TARGET_NUMBER_OF_HOLDINGS
        ):
            break

        code = row["baostock_code"]
        industry = row["industry"]

        if code in selected_codes:
            continue

        if not can_add_industry(
            industry,
            industry_counts,
        ):
            continue

        selected_records.append(
            create_selected_record(
                row=row,
                selection_status="RETAINED",
                selection_reason=(
                    "RANK_WITHIN_EXIT_BUFFER"
                ),
            )
        )

        selected_codes.add(code)
        industry_counts[industry] += 1

    # Step 2: add new stocks ranked in the top 30.
    entry_pool = candidates.loc[
        candidates["walk_forward_rank"]
        <= ENTRY_RANK_LIMIT
    ].sort_values(
        "walk_forward_rank"
    )

    for _, row in entry_pool.iterrows():
        if (
            len(selected_records)
            >= TARGET_NUMBER_OF_HOLDINGS
        ):
            break

        code = row["baostock_code"]
        industry = row["industry"]

        if code in selected_codes:
            continue

        if not can_add_industry(
            industry,
            industry_counts,
        ):
            continue

        if code in previous_codes:
            selection_status = "RETAINED"
            selection_reason = (
                "REENTERED_TOP_30"
            )
        else:
            selection_status = "NEW_ENTRY"
            selection_reason = (
                "RANK_WITHIN_ENTRY_LIMIT"
            )

        selected_records.append(
            create_selected_record(
                row=row,
                selection_status=(
                    selection_status
                ),
                selection_reason=(
                    selection_reason
                ),
            )
        )

        selected_codes.add(code)
        industry_counts[industry] += 1

    if not selected_records:
        return pd.DataFrame()

    selected = pd.DataFrame(
        selected_records
    ).sort_values(
        "walk_forward_rank"
    ).reset_index(drop=True)

    selected["portfolio_position"] = (
        np.arange(len(selected)) + 1
    )

    selected[
        "industry_position_number"
    ] = selected.groupby(
        "industry"
    ).cumcount() + 1

    selected["invested_weight"] = (
        selected["target_weight"]
    )

    return selected


def determine_sell_reason(
    code: str,
    current_candidates: pd.DataFrame,
) -> str:
    current_row = current_candidates.loc[
        current_candidates[
            "baostock_code"
        ]
        == code
    ]

    if current_row.empty:
        return "NO_LONGER_ELIGIBLE"

    current_rank = current_row[
        "walk_forward_rank"
    ].iloc[0]

    if current_rank > EXIT_RANK_LIMIT:
        return "RANK_BELOW_EXIT_LIMIT"

    return "INDUSTRY_OR_PORTFOLIO_LIMIT"


def build_rebalance_orders(
    snapshot_date: pd.Timestamp,
    current_candidates: pd.DataFrame,
    previous_holdings: dict[str, dict],
    target_holdings: pd.DataFrame,
) -> pd.DataFrame:
    if target_holdings.empty:
        target_lookup = {}
    else:
        target_lookup = {
            row["baostock_code"]: row
            for _, row in (
                target_holdings.iterrows()
            )
        }

    candidate_lookup = {
        row["baostock_code"]: row
        for _, row in (
            current_candidates.iterrows()
        )
    }

    all_codes = sorted(
        set(previous_holdings)
        | set(target_lookup)
    )

    order_rows = []

    for code in all_codes:
        previous_record = (
            previous_holdings.get(code)
        )
        target_record = (
            target_lookup.get(code)
        )
        candidate_record = (
            candidate_lookup.get(code)
        )

        current_weight = (
            float(
                previous_record[
                    "target_weight"
                ]
            )
            if previous_record is not None
            else 0.0
        )

        target_weight = (
            float(
                target_record[
                    "target_weight"
                ]
            )
            if target_record is not None
            else 0.0
        )

        weight_change = (
            target_weight - current_weight
        )

        if (
            current_weight <= WEIGHT_TOLERANCE
            and target_weight
            > WEIGHT_TOLERANCE
        ):
            action = "BUY"
            action_reason = (
                target_record[
                    "selection_reason"
                ]
            )

        elif (
            current_weight
            > WEIGHT_TOLERANCE
            and target_weight
            <= WEIGHT_TOLERANCE
        ):
            action = "SELL"
            action_reason = determine_sell_reason(
                code=code,
                current_candidates=(
                    current_candidates
                ),
            )

        elif abs(weight_change) <= (
            WEIGHT_TOLERANCE
        ):
            action = "HOLD"
            action_reason = (
                "POSITION_RETAINED"
            )

        elif weight_change > 0:
            action = "INCREASE"
            action_reason = (
                "TARGET_WEIGHT_INCREASED"
            )

        else:
            action = "REDUCE"
            action_reason = (
                "TARGET_WEIGHT_REDUCED"
            )

        information_record = (
            target_record
            if target_record is not None
            else candidate_record
        )

        if information_record is None:
            information_record = (
                previous_record
            )

        order_rows.append(
            {
                "snapshot_date": (
                    snapshot_date
                ),
                "signal_price_date": (
                    information_record.get(
                        "signal_price_date",
                        pd.NaT,
                    )
                ),
                "baostock_code": code,
                "symbol": (
                    information_record.get(
                        "symbol",
                        None,
                    )
                ),
                "stock_name": (
                    information_record.get(
                        "stock_name",
                        None,
                    )
                ),
                "industry": (
                    information_record.get(
                        "industry",
                        "UNKNOWN",
                    )
                ),
                "walk_forward_rank": (
                    information_record.get(
                        "walk_forward_rank",
                        np.nan,
                    )
                ),
                "walk_forward_score": (
                    information_record.get(
                        "walk_forward_score",
                        np.nan,
                    )
                ),
                "current_weight": (
                    current_weight
                ),
                "target_weight": (
                    target_weight
                ),
                "weight_change": (
                    weight_change
                ),
                "absolute_weight_change": (
                    abs(weight_change)
                ),
                "action": action,
                "action_reason": action_reason,
            }
        )

    return pd.DataFrame(order_rows)


def validate_target_portfolio(
    snapshot_date: pd.Timestamp,
    target_holdings: pd.DataFrame,
) -> None:
    if target_holdings.empty:
        return

    duplicate_count = int(
        target_holdings[
            "baostock_code"
        ].duplicated().sum()
    )

    if duplicate_count:
        raise ValueError(
            f"{snapshot_date.date()}: "
            f"duplicate holdings detected."
        )

    if len(target_holdings) > (
        TARGET_NUMBER_OF_HOLDINGS
    ):
        raise ValueError(
            f"{snapshot_date.date()}: "
            f"too many holdings."
        )

    maximum_weight = target_holdings[
        "target_weight"
    ].max()

    if maximum_weight > (
        TARGET_STOCK_WEIGHT
        + WEIGHT_TOLERANCE
    ):
        raise ValueError(
            f"{snapshot_date.date()}: "
            f"stock weight limit exceeded."
        )

    total_weight = target_holdings[
        "target_weight"
    ].sum()

    if total_weight > (
        1.0 + WEIGHT_TOLERANCE
    ):
        raise ValueError(
            f"{snapshot_date.date()}: "
            f"portfolio weight exceeds 100%."
        )

    maximum_industry_count = (
        target_holdings.groupby(
            "industry"
        ).size().max()
    )

    if maximum_industry_count > (
        MAXIMUM_STOCKS_PER_INDUSTRY
    ):
        raise ValueError(
            f"{snapshot_date.date()}: "
            f"industry holding limit exceeded."
        )

    new_entries = target_holdings.loc[
        target_holdings[
            "selection_status"
        ]
        == "NEW_ENTRY"
    ]

    invalid_entries = new_entries.loc[
        new_entries[
            "walk_forward_rank"
        ]
        > ENTRY_RANK_LIMIT
    ]

    if not invalid_entries.empty:
        raise ValueError(
            f"{snapshot_date.date()}: "
            f"entry-rank limit exceeded."
        )


def create_snapshot_summary(
    snapshot_date: pd.Timestamp,
    target_holdings: pd.DataFrame,
    orders: pd.DataFrame,
) -> dict:
    if target_holdings.empty:
        number_of_holdings = 0
        invested_weight = 0.0
        number_of_industries = 0
        maximum_industry_holdings = 0
        retained_holdings = 0
        new_entries = 0
    else:
        number_of_holdings = len(
            target_holdings
        )

        invested_weight = float(
            target_holdings[
                "target_weight"
            ].sum()
        )

        industry_counts = (
            target_holdings.groupby(
                "industry"
            ).size()
        )

        number_of_industries = len(
            industry_counts
        )

        maximum_industry_holdings = int(
            industry_counts.max()
        )

        retained_holdings = int(
            (
                target_holdings[
                    "selection_status"
                ]
                == "RETAINED"
            ).sum()
        )

        new_entries = int(
            (
                target_holdings[
                    "selection_status"
                ]
                == "NEW_ENTRY"
            ).sum()
        )

    buy_weight = float(
        orders.loc[
            orders["weight_change"] > 0,
            "weight_change",
        ].sum()
    )

    sell_weight = float(
        -orders.loc[
            orders["weight_change"] < 0,
            "weight_change",
        ].sum()
    )

    gross_traded_weight = float(
        orders[
            "absolute_weight_change"
        ].sum()
    )

    one_way_turnover = (
        gross_traded_weight / 2.0
    )

    return {
        "snapshot_date": snapshot_date,
        "number_of_holdings": (
            number_of_holdings
        ),
        "target_number_of_holdings": (
            TARGET_NUMBER_OF_HOLDINGS
        ),
        "number_of_industries": (
            number_of_industries
        ),
        "maximum_industry_holdings": (
            maximum_industry_holdings
        ),
        "retained_holdings": (
            retained_holdings
        ),
        "new_entries": new_entries,
        "buy_orders": int(
            (orders["action"] == "BUY").sum()
        ),
        "sell_orders": int(
            (orders["action"] == "SELL").sum()
        ),
        "hold_orders": int(
            (orders["action"] == "HOLD").sum()
        ),
        "invested_weight": invested_weight,
        "cash_weight": max(
            0.0,
            1.0 - invested_weight,
        ),
        "buy_weight": buy_weight,
        "sell_weight": sell_weight,
        "gross_traded_weight": (
            gross_traded_weight
        ),
        "one_way_turnover": (
            one_way_turnover
        ),
        "portfolio_status": (
            "FULLY_INVESTED"
            if number_of_holdings
            == TARGET_NUMBER_OF_HOLDINGS
            else "PARTIAL_PORTFOLIO"
        ),
    }


def construct_all_portfolios(
    scores: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    target_frames = []
    order_frames = []
    summary_rows = []

    previous_holdings: dict[str, dict] = {}

    grouped_snapshots = list(
        scores.groupby(
            "snapshot_date",
            sort=True,
        )
    )

    total_snapshots = len(
        grouped_snapshots
    )

    for position, (
        snapshot_date,
        candidates,
    ) in enumerate(
        grouped_snapshots,
        start=1,
    ):
        candidates = candidates.sort_values(
            "walk_forward_rank"
        ).copy()

        target_holdings = (
            select_monthly_portfolio(
                candidates=candidates,
                previous_holdings=(
                    previous_holdings
                ),
            )
        )

        validate_target_portfolio(
            snapshot_date=snapshot_date,
            target_holdings=(
                target_holdings
            ),
        )

        orders = build_rebalance_orders(
            snapshot_date=snapshot_date,
            current_candidates=candidates,
            previous_holdings=(
                previous_holdings
            ),
            target_holdings=target_holdings,
        )

        summary = create_snapshot_summary(
            snapshot_date=snapshot_date,
            target_holdings=(
                target_holdings
            ),
            orders=orders,
        )

        if not target_holdings.empty:
            target_frames.append(
                target_holdings
            )

        if not orders.empty:
            order_frames.append(orders)

        summary_rows.append(summary)

        previous_holdings = {
            row["baostock_code"]: (
                row.to_dict()
            )
            for _, row in (
                target_holdings.iterrows()
            )
        }

        if (
            position == 1
            or position % 10 == 0
            or position == total_snapshots
        ):
            print(
                f"[{position}/"
                f"{total_snapshots}] "
                f"Portfolio built for "
                f"{snapshot_date.date()}: "
                f"{len(target_holdings)} holdings",
                flush=True,
            )

    targets = pd.concat(
        target_frames,
        ignore_index=True,
    )

    orders = pd.concat(
        order_frames,
        ignore_index=True,
    )

    summary = pd.DataFrame(
        summary_rows
    )

    return targets, orders, summary


def format_dates(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    output = frame.copy()

    for column in output.columns:
        if (
            column.endswith("_date")
            or column == "snapshot_date"
        ):
            output[column] = pd.to_datetime(
                output[column],
                errors="coerce",
            ).dt.strftime("%Y-%m-%d")

    return output


def save_outputs(
    connection: sqlite3.Connection,
    targets: pd.DataFrame,
    orders: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_items = [
        (
            TARGET_TABLE,
            format_dates(targets),
            TARGET_FILE,
        ),
        (
            ORDER_TABLE,
            format_dates(orders),
            ORDER_FILE,
        ),
        (
            SUMMARY_TABLE,
            format_dates(summary),
            SUMMARY_FILE,
        ),
    ]

    for (
        table_name,
        frame,
        output_file,
    ) in output_items:
        frame.to_sql(
            table_name,
            connection,
            if_exists="replace",
            index=False,
            chunksize=2_000,
        )

        frame.to_csv(
            output_file,
            index=False,
            encoding="utf-8-sig",
            float_format="%.10f",
        )

    connection.execute(
        f"""
        CREATE UNIQUE INDEX IF NOT EXISTS
        idx_{TARGET_TABLE}_snapshot_code
        ON {TARGET_TABLE} (
            snapshot_date,
            baostock_code
        );
        """
    )

    connection.execute(
        f"""
        CREATE INDEX IF NOT EXISTS
        idx_{ORDER_TABLE}_snapshot_action
        ON {ORDER_TABLE} (
            snapshot_date,
            action
        );
        """
    )

    connection.commit()


def print_latest_portfolio(
    targets: pd.DataFrame,
) -> None:
    latest_snapshot = targets[
        "snapshot_date"
    ].max()

    latest = targets.loc[
        targets["snapshot_date"]
        == latest_snapshot
    ].sort_values(
        "portfolio_position"
    )

    display_columns = [
        "portfolio_position",
        "symbol",
        "stock_name",
        "industry",
        "walk_forward_rank",
        "walk_forward_score",
        "selection_status",
        "target_weight",
    ]

    print()
    print("=" * 140)
    print(
        f"LATEST TARGET PORTFOLIO: "
        f"{latest_snapshot.date()}"
    )
    print("=" * 140)

    print(
        latest[display_columns].to_string(
            index=False,
            formatters={
                "walk_forward_score": (
                    lambda value: f"{value:.4f}"
                ),
                "target_weight": (
                    lambda value: f"{value:.2%}"
                ),
            },
        )
    )


def print_latest_trades(
    orders: pd.DataFrame,
) -> None:
    latest_snapshot = orders[
        "snapshot_date"
    ].max()

    latest = orders.loc[
        (
            orders["snapshot_date"]
            == latest_snapshot
        )
        & (
            orders["action"].isin(
                ["BUY", "SELL"]
            )
        )
    ].copy()

    latest = latest.sort_values(
        [
            "action",
            "walk_forward_rank",
        ]
    )

    display_columns = [
        "action",
        "symbol",
        "stock_name",
        "industry",
        "walk_forward_rank",
        "current_weight",
        "target_weight",
        "action_reason",
    ]

    print()
    print("=" * 140)
    print(
        f"LATEST BUY/SELL ORDERS: "
        f"{latest_snapshot.date()}"
    )
    print("=" * 140)

    if latest.empty:
        print("No buy or sell orders.")
        return

    print(
        latest[display_columns].to_string(
            index=False,
            formatters={
                "current_weight": (
                    lambda value: f"{value:.2%}"
                ),
                "target_weight": (
                    lambda value: f"{value:.2%}"
                ),
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
        240,
    )

    database_path = resolve_database_path()

    print("=" * 140)
    print("V2 PORTFOLIO CONSTRUCTION")
    print("=" * 140)
    print(f"Database: {database_path}")
    print(
        f"Target holdings: "
        f"{TARGET_NUMBER_OF_HOLDINGS}"
    )
    print(
        f"Target stock weight: "
        f"{TARGET_STOCK_WEIGHT:.2%}"
    )
    print(
        f"Maximum stocks per industry: "
        f"{MAXIMUM_STOCKS_PER_INDUSTRY}"
    )
    print(
        f"Entry rank limit: "
        f"{ENTRY_RANK_LIMIT}"
    )
    print(
        f"Exit rank limit: "
        f"{EXIT_RANK_LIMIT}"
    )
    print()

    with sqlite3.connect(
        database_path
    ) as connection:
        if not table_exists(
            connection,
            SCORE_TABLE,
        ):
            raise RuntimeError(
                f"Required table does not exist: "
                f"{SCORE_TABLE}"
            )

        print(
            "Loading walk-forward scores...",
            flush=True,
        )

        scores = load_walk_forward_scores(
            connection
        )

        print(
            f"Model-ready snapshots: "
            f"{scores['snapshot_date'].nunique():,}"
        )
        print(
            f"Candidate observations: "
            f"{len(scores):,}"
        )

        print(
            "Constructing monthly portfolios...",
            flush=True,
        )

        (
            targets,
            orders,
            summary,
        ) = construct_all_portfolios(scores)

        constraint_violations = int(
            (
                summary[
                    "maximum_industry_holdings"
                ]
                > MAXIMUM_STOCKS_PER_INDUSTRY
            ).sum()
        )

        partial_portfolios = int(
            (
                summary["portfolio_status"]
                == "PARTIAL_PORTFOLIO"
            ).sum()
        )

        later_summary = summary.iloc[
            1:
        ].copy()

        average_one_way_turnover = (
            later_summary[
                "one_way_turnover"
            ].mean()
            if not later_summary.empty
            else np.nan
        )

        print()
        print("=" * 140)
        print("PORTFOLIO CONSTRUCTION SUMMARY")
        print("=" * 140)
        print(
            f"Portfolio snapshots: "
            f"{len(summary):,}"
        )
        print(
            f"Target holding observations: "
            f"{len(targets):,}"
        )
        print(
            f"Average holdings: "
            f"{summary['number_of_holdings'].mean():.2f}"
        )
        print(
            f"Average monthly one-way turnover "
            f"excluding initial portfolio: "
            f"{average_one_way_turnover:.2%}"
        )
        print(
            f"Partial portfolios: "
            f"{partial_portfolios}"
        )
        print(
            f"Industry-limit violations: "
            f"{constraint_violations}"
        )
        print(
            f"Maximum total portfolio weight: "
            f"{summary['invested_weight'].max():.2%}"
        )

        if constraint_violations:
            raise RuntimeError(
                "Portfolio constraint audit failed."
            )

        print(
            "Saving portfolio targets "
            "and orders...",
            flush=True,
        )

        save_outputs(
            connection=connection,
            targets=targets,
            orders=orders,
            summary=summary,
        )

    print_latest_portfolio(targets)
    print_latest_trades(orders)

    print()
    print("=" * 140)
    print("OUTPUT FILES")
    print("=" * 140)
    print(TARGET_FILE)
    print(ORDER_FILE)
    print(SUMMARY_FILE)
    print()
    print(
        "V2 portfolio construction completed."
    )


if __name__ == "__main__":
    main()