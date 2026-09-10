"""
V2 event-driven portfolio backtester.

Features:

- Signals formed after month-end close
- Trades executed at the next trading-day open
- Sell orders executed before buy orders
- Unadjusted prices used for execution and board-lot calculation
- Daily pct_change/pre-close data used for economic position returns
- Suspension and one-price limit checks
- 100-share board-lot approximation
- Commission, minimum commission, transfer fee, stamp duty and slippage
- Cash cannot become negative
- Actual holdings may differ from targets after failed orders
- Realistic-cost and zero-cost comparison
- CSI 300 benchmark
"""

from __future__ import annotations

import math
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from src.factors_v2 import resolve_database_path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "reports" / "results"

TARGET_TABLE = "portfolio_targets_v2"
PRICE_TABLE = "execution_prices_v2"

DAILY_TABLE = "event_backtest_daily_v2"
TRADE_TABLE = "event_backtest_trades_v2"
POSITION_TABLE = "event_backtest_positions_v2"
AUDIT_TABLE = "event_backtest_rebalance_audit_v2"
PERFORMANCE_TABLE = "event_backtest_performance_v2"

DAILY_FILE = RESULTS_DIR / "event_backtest_daily_v2.csv"
TRADE_FILE = RESULTS_DIR / "event_backtest_trades_v2.csv"
POSITION_FILE = RESULTS_DIR / "event_backtest_positions_v2.csv"
AUDIT_FILE = RESULTS_DIR / "event_backtest_rebalance_audit_v2.csv"
PERFORMANCE_FILE = RESULTS_DIR / "event_backtest_performance_v2.csv"

BENCHMARK_CODE = "sh.000300"

INITIAL_CAPITAL = 1_000_000.0
RISK_FREE_RATE = 0.02

BOARD_LOT = 100

COMMISSION_RATE = 0.0003
MINIMUM_COMMISSION = 5.0
SLIPPAGE_RATE = 0.0005

STAMP_DUTY_CHANGE_DATE = pd.Timestamp("2023-08-28")
STAMP_DUTY_RATE_BEFORE = 0.001
STAMP_DUTY_RATE_AFTER = 0.0005

TRANSFER_FEE_CHANGE_DATE = pd.Timestamp("2022-04-29")
TRANSFER_FEE_RATE_BEFORE = 0.00002
TRANSFER_FEE_RATE_AFTER = 0.00001

POSITION_TOLERANCE = 0.01


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


def load_targets(
    connection: sqlite3.Connection,
) -> pd.DataFrame:
    targets = pd.read_sql_query(
        f'SELECT * FROM "{TARGET_TABLE}"',
        connection,
    )

    required_columns = {
        "snapshot_date",
        "signal_price_date",
        "baostock_code",
        "target_weight",
    }

    missing = required_columns.difference(
        targets.columns
    )

    if missing:
        raise ValueError(
            f"{TARGET_TABLE} is missing columns: "
            f"{sorted(missing)}"
        )

    targets["snapshot_date"] = pd.to_datetime(
        targets["snapshot_date"],
        errors="coerce",
    )

    targets["signal_price_date"] = pd.to_datetime(
        targets["signal_price_date"],
        errors="coerce",
    )

    targets["target_weight"] = pd.to_numeric(
        targets["target_weight"],
        errors="coerce",
    )

    if "portfolio_position" in targets.columns:
        targets["portfolio_position"] = (
            pd.to_numeric(
                targets["portfolio_position"],
                errors="coerce",
            )
        )
    else:
        targets["portfolio_position"] = (
            targets.groupby(
                "snapshot_date"
            ).cumcount() + 1
        )

    targets = targets.dropna(
        subset=[
            "snapshot_date",
            "signal_price_date",
            "baostock_code",
            "target_weight",
        ]
    )

    return targets.sort_values(
        [
            "snapshot_date",
            "portfolio_position",
        ]
    ).reset_index(drop=True)


def load_execution_prices(
    connection: sqlite3.Connection,
) -> pd.DataFrame:
    prices = pd.read_sql_query(
        f'SELECT * FROM "{PRICE_TABLE}"',
        connection,
    )

    required_columns = {
        "date",
        "baostock_code",
        "open",
        "high",
        "low",
        "close",
        "pre_close",
        "trade_status",
        "pct_change",
        "is_st",
    }

    missing = required_columns.difference(
        prices.columns
    )

    if missing:
        raise ValueError(
            f"{PRICE_TABLE} is missing columns: "
            f"{sorted(missing)}"
        )

    prices["date"] = pd.to_datetime(
        prices["date"],
        errors="coerce",
    )

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
        if column in prices.columns:
            prices[column] = pd.to_numeric(
                prices[column],
                errors="coerce",
            )

    prices = prices.dropna(
        subset=[
            "date",
            "baostock_code",
        ]
    )

    prices = prices.drop_duplicates(
        ["date", "baostock_code"],
        keep="last",
    )

    return prices.sort_values(
        ["date", "baostock_code"]
    ).reset_index(drop=True)


def build_trading_calendar(
    prices: pd.DataFrame,
) -> list[pd.Timestamp]:
    benchmark = prices.loc[
        (
            prices["baostock_code"]
            == BENCHMARK_CODE
        )
        & (
            prices["trade_status"] == 1
        )
        & prices["close"].notna()
    ]

    if benchmark.empty:
        raise ValueError(
            "CSI 300 benchmark prices are missing."
        )

    return sorted(
        benchmark["date"].unique()
    )


def build_bar_lookup(
    prices: pd.DataFrame,
) -> dict[pd.Timestamp, pd.DataFrame]:
    return {
        pd.Timestamp(date): group.set_index(
            "baostock_code",
            drop=False,
        )
        for date, group in prices.groupby(
            "date",
            sort=True,
        )
    }


def get_bar(
    bar_lookup: dict[pd.Timestamp, pd.DataFrame],
    date: pd.Timestamp,
    code: str,
) -> pd.Series | None:
    day_bars = bar_lookup.get(
        pd.Timestamp(date)
    )

    if day_bars is None:
        return None

    if code not in day_bars.index:
        return None

    bar = day_bars.loc[code]

    if isinstance(bar, pd.DataFrame):
        bar = bar.iloc[-1]

    return bar


def build_execution_schedule(
    targets: pd.DataFrame,
    trading_calendar: list[pd.Timestamp],
) -> tuple[dict[pd.Timestamp, pd.DataFrame], list[dict]]:
    calendar_array = np.array(
        trading_calendar,
        dtype="datetime64[ns]",
    )

    schedule: dict[pd.Timestamp, pd.DataFrame] = {}
    unscheduled_rows = []

    for snapshot_date, snapshot in targets.groupby(
        "snapshot_date",
        sort=True,
    ):
        signal_date = snapshot[
            "signal_price_date"
        ].max()

        position = np.searchsorted(
            calendar_array,
            np.datetime64(signal_date),
            side="right",
        )

        if position >= len(calendar_array):
            unscheduled_rows.append(
                {
                    "snapshot_date": snapshot_date,
                    "signal_price_date": signal_date,
                    "execution_date": pd.NaT,
                    "status": "NO_FUTURE_TRADING_DATE",
                }
            )
            continue

        execution_date = pd.Timestamp(
            calendar_array[position]
        )

        schedule[execution_date] = (
            snapshot.copy()
        )

    return schedule, unscheduled_rows


def price_limit_ratio(
    code: str,
    date: pd.Timestamp,
    is_st: int,
) -> float:
    if is_st == 1:
        return 0.05

    if code.startswith("sh.688"):
        return 0.20

    if (
        code.startswith("sz.300")
        or code.startswith("sz.301")
    ):
        if date >= pd.Timestamp(
            "2020-08-24"
        ):
            return 0.20

    return 0.10


def tradability_status(
    bar: pd.Series | None,
    side: str,
    date: pd.Timestamp,
    code: str,
) -> tuple[bool, str]:
    if bar is None:
        return False, "NO_EXECUTION_BAR"

    if (
        pd.isna(bar["trade_status"])
        or int(bar["trade_status"]) != 1
    ):
        return False, "SUSPENDED"

    required_prices = [
        bar["open"],
        bar["high"],
        bar["low"],
        bar["close"],
        bar["pre_close"],
    ]

    if any(
        pd.isna(value) or value <= 0
        for value in required_prices
    ):
        return False, "INVALID_EXECUTION_PRICE"

    if (
        "volume" in bar.index
        and pd.notna(bar["volume"])
        and bar["volume"] <= 0
    ):
        return False, "ZERO_VOLUME"

    tolerance = max(
        0.001,
        float(bar["pre_close"]) * 0.0005,
    )

    one_price_market = (
        abs(
            float(bar["high"])
            - float(bar["low"])
        )
        <= tolerance
    )

    daily_change = (
        float(bar["close"])
        / float(bar["pre_close"])
        - 1.0
    )

    limit_ratio = price_limit_ratio(
        code=code,
        date=date,
        is_st=int(
            bar["is_st"]
            if pd.notna(bar["is_st"])
            else 0
        ),
    )

    if (
        side == "BUY"
        and one_price_market
        and daily_change
        >= limit_ratio - 0.002
    ):
        return False, "ONE_PRICE_LIMIT_UP"

    if (
        side == "SELL"
        and one_price_market
        and daily_change
        <= -limit_ratio + 0.002
    ):
        return False, "ONE_PRICE_LIMIT_DOWN"

    return True, "TRADABLE"


def stamp_duty_rate(
    date: pd.Timestamp,
) -> float:
    if date < STAMP_DUTY_CHANGE_DATE:
        return STAMP_DUTY_RATE_BEFORE

    return STAMP_DUTY_RATE_AFTER


def transfer_fee_rate(
    date: pd.Timestamp,
) -> float:
    if date < TRANSFER_FEE_CHANGE_DATE:
        return TRANSFER_FEE_RATE_BEFORE

    return TRANSFER_FEE_RATE_AFTER


def calculate_explicit_costs(
    trade_value: float,
    side: str,
    date: pd.Timestamp,
    apply_costs: bool,
) -> tuple[float, float, float]:
    if not apply_costs:
        return 0.0, 0.0, 0.0

    commission = max(
        trade_value * COMMISSION_RATE,
        MINIMUM_COMMISSION,
    )

    transfer_fee = (
        trade_value
        * transfer_fee_rate(date)
    )

    stamp_duty = 0.0

    if side == "SELL":
        stamp_duty = (
            trade_value
            * stamp_duty_rate(date)
        )

    return (
        commission,
        transfer_fee,
        stamp_duty,
    )


def update_positions_to_open(
    positions: dict[str, float],
    date: pd.Timestamp,
    bar_lookup: dict[pd.Timestamp, pd.DataFrame],
) -> None:
    for code in list(positions):
        bar = get_bar(
            bar_lookup,
            date,
            code,
        )

        if bar is None:
            continue

        if (
            pd.notna(bar["open"])
            and pd.notna(bar["pre_close"])
            and bar["open"] > 0
            and bar["pre_close"] > 0
        ):
            positions[code] *= (
                float(bar["open"])
                / float(bar["pre_close"])
            )


def update_positions_to_close(
    positions: dict[str, float],
    date: pd.Timestamp,
    bar_lookup: dict[pd.Timestamp, pd.DataFrame],
) -> None:
    for code in list(positions):
        bar = get_bar(
            bar_lookup,
            date,
            code,
        )

        if bar is not None:
            if (
                pd.notna(bar["close"])
                and pd.notna(bar["open"])
                and bar["close"] > 0
                and bar["open"] > 0
            ):
                positions[code] *= (
                    float(bar["close"])
                    / float(bar["open"])
                )

        if positions[code] <= POSITION_TOLERANCE:
            positions.pop(code, None)


def make_trade_record(
    snapshot_date: pd.Timestamp,
    execution_date: pd.Timestamp,
    code: str,
    target_record: dict | None,
    side: str,
    status: str,
    status_reason: str,
    requested_value: float,
    filled_mid_value: float,
    filled_shares: int,
    market_open_price: float | None,
    execution_price: float | None,
    commission: float,
    transfer_fee: float,
    stamp_duty: float,
    slippage_cost: float,
    cash_after: float,
) -> dict:
    return {
        "snapshot_date": snapshot_date,
        "execution_date": execution_date,
        "baostock_code": code,
        "symbol": (
            target_record.get("symbol")
            if target_record
            else None
        ),
        "stock_name": (
            target_record.get("stock_name")
            if target_record
            else None
        ),
        "industry": (
            target_record.get("industry")
            if target_record
            else None
        ),
        "side": side,
        "status": status,
        "status_reason": status_reason,
        "requested_value": requested_value,
        "filled_mid_value": filled_mid_value,
        "filled_shares": filled_shares,
        "market_open_price": market_open_price,
        "execution_price": execution_price,
        "commission": commission,
        "transfer_fee": transfer_fee,
        "stamp_duty": stamp_duty,
        "slippage_cost": slippage_cost,
        "total_cost": (
            commission
            + transfer_fee
            + stamp_duty
            + slippage_cost
        ),
        "cash_after": cash_after,
    }


def execute_rebalance(
    snapshot: pd.DataFrame,
    execution_date: pd.Timestamp,
    positions: dict[str, float],
    cash: float,
    bar_lookup: dict[pd.Timestamp, pd.DataFrame],
    apply_costs: bool,
) -> tuple[
    dict[str, float],
    float,
    list[dict],
    dict,
]:
    snapshot_date = snapshot[
        "snapshot_date"
    ].iloc[0]

    target_records = {
        row["baostock_code"]: (
            row.to_dict()
        )
        for _, row in snapshot.iterrows()
    }

    target_weights = {
        code: float(
            record["target_weight"]
        )
        for code, record in (
            target_records.items()
        )
    }

    portfolio_value_before = (
        cash + sum(positions.values())
    )

    trade_rows = []
    executed_sells = 0
    executed_buys = 0
    blocked_orders = 0

    # Sell before buying.
    sell_codes = sorted(
        set(positions)
    )

    for code in sell_codes:
        current_value = positions.get(
            code,
            0.0,
        )

        desired_value = (
            portfolio_value_before
            * target_weights.get(code, 0.0)
        )

        requested_value = (
            current_value - desired_value
        )

        if requested_value <= POSITION_TOLERANCE:
            continue

        bar = get_bar(
            bar_lookup,
            execution_date,
            code,
        )

        tradable, reason = tradability_status(
            bar=bar,
            side="SELL",
            date=execution_date,
            code=code,
        )

        target_record = target_records.get(
            code
        )

        if not tradable:
            blocked_orders += 1

            trade_rows.append(
                make_trade_record(
                    snapshot_date=snapshot_date,
                    execution_date=execution_date,
                    code=code,
                    target_record=target_record,
                    side="SELL",
                    status="BLOCKED",
                    status_reason=reason,
                    requested_value=(
                        requested_value
                    ),
                    filled_mid_value=0.0,
                    filled_shares=0,
                    market_open_price=(
                        float(bar["open"])
                        if (
                            bar is not None
                            and pd.notna(
                                bar["open"]
                            )
                        )
                        else None
                    ),
                    execution_price=None,
                    commission=0.0,
                    transfer_fee=0.0,
                    stamp_duty=0.0,
                    slippage_cost=0.0,
                    cash_after=cash,
                )
            )
            continue

        open_price = float(bar["open"])

        if desired_value <= POSITION_TOLERANCE:
            filled_mid_value = current_value
            filled_shares = max(
                1,
                int(round(
                    current_value
                    / open_price
                )),
            )
        else:
            filled_shares = int(
                math.floor(
                    requested_value
                    / open_price
                    / BOARD_LOT
                )
                * BOARD_LOT
            )

            filled_mid_value = min(
                current_value,
                filled_shares
                * open_price,
            )

        if (
            filled_shares <= 0
            or filled_mid_value
            <= POSITION_TOLERANCE
        ):
            trade_rows.append(
                make_trade_record(
                    snapshot_date=snapshot_date,
                    execution_date=execution_date,
                    code=code,
                    target_record=target_record,
                    side="SELL",
                    status="SKIPPED",
                    status_reason=(
                        "BELOW_BOARD_LOT"
                    ),
                    requested_value=(
                        requested_value
                    ),
                    filled_mid_value=0.0,
                    filled_shares=0,
                    market_open_price=open_price,
                    execution_price=None,
                    commission=0.0,
                    transfer_fee=0.0,
                    stamp_duty=0.0,
                    slippage_cost=0.0,
                    cash_after=cash,
                )
            )
            continue

        applied_slippage = (
            SLIPPAGE_RATE
            if apply_costs
            else 0.0
        )

        execution_price = (
            open_price
            * (1.0 - applied_slippage)
        )

        execution_value = (
            filled_mid_value
            * (1.0 - applied_slippage)
        )

        (
            commission,
            transfer_fee,
            stamp_duty,
        ) = calculate_explicit_costs(
            trade_value=execution_value,
            side="SELL",
            date=execution_date,
            apply_costs=apply_costs,
        )

        slippage_cost = (
            filled_mid_value
            - execution_value
        )

        cash += (
            execution_value
            - commission
            - transfer_fee
            - stamp_duty
        )

        positions[code] = max(
            0.0,
            current_value
            - filled_mid_value,
        )

        if positions[code] <= POSITION_TOLERANCE:
            positions.pop(code, None)

        executed_sells += 1

        trade_rows.append(
            make_trade_record(
                snapshot_date=snapshot_date,
                execution_date=execution_date,
                code=code,
                target_record=target_record,
                side="SELL",
                status="EXECUTED",
                status_reason="FILLED_AT_OPEN",
                requested_value=(
                    requested_value
                ),
                filled_mid_value=(
                    filled_mid_value
                ),
                filled_shares=(
                    filled_shares
                ),
                market_open_price=open_price,
                execution_price=(
                    execution_price
                ),
                commission=commission,
                transfer_fee=transfer_fee,
                stamp_duty=stamp_duty,
                slippage_cost=slippage_cost,
                cash_after=cash,
            )
        )

    # Buy after sale proceeds become available.
    ordered_targets = snapshot.sort_values(
        "portfolio_position"
    )

    for _, target_row in (
        ordered_targets.iterrows()
    ):
        code = target_row[
            "baostock_code"
        ]

        target_record = (
            target_row.to_dict()
        )

        desired_value = (
            portfolio_value_before
            * float(
                target_row[
                    "target_weight"
                ]
            )
        )

        current_value = positions.get(
            code,
            0.0,
        )

        requested_value = (
            desired_value - current_value
        )

        if requested_value <= POSITION_TOLERANCE:
            continue

        bar = get_bar(
            bar_lookup,
            execution_date,
            code,
        )

        tradable, reason = tradability_status(
            bar=bar,
            side="BUY",
            date=execution_date,
            code=code,
        )

        if not tradable:
            blocked_orders += 1

            trade_rows.append(
                make_trade_record(
                    snapshot_date=snapshot_date,
                    execution_date=execution_date,
                    code=code,
                    target_record=target_record,
                    side="BUY",
                    status="BLOCKED",
                    status_reason=reason,
                    requested_value=(
                        requested_value
                    ),
                    filled_mid_value=0.0,
                    filled_shares=0,
                    market_open_price=(
                        float(bar["open"])
                        if (
                            bar is not None
                            and pd.notna(
                                bar["open"]
                            )
                        )
                        else None
                    ),
                    execution_price=None,
                    commission=0.0,
                    transfer_fee=0.0,
                    stamp_duty=0.0,
                    slippage_cost=0.0,
                    cash_after=cash,
                )
            )
            continue

        open_price = float(bar["open"])

        filled_shares = int(
            math.floor(
                requested_value
                / open_price
                / BOARD_LOT
            )
            * BOARD_LOT
        )

        if filled_shares <= 0:
            trade_rows.append(
                make_trade_record(
                    snapshot_date=snapshot_date,
                    execution_date=execution_date,
                    code=code,
                    target_record=target_record,
                    side="BUY",
                    status="SKIPPED",
                    status_reason=(
                        "BELOW_BOARD_LOT"
                    ),
                    requested_value=(
                        requested_value
                    ),
                    filled_mid_value=0.0,
                    filled_shares=0,
                    market_open_price=open_price,
                    execution_price=None,
                    commission=0.0,
                    transfer_fee=0.0,
                    stamp_duty=0.0,
                    slippage_cost=0.0,
                    cash_after=cash,
                )
            )
            continue

        applied_slippage = (
            SLIPPAGE_RATE
            if apply_costs
            else 0.0
        )

        while filled_shares > 0:
            execution_price = (
                open_price
                * (1.0 + applied_slippage)
            )

            execution_value = (
                filled_shares
                * execution_price
            )

            (
                commission,
                transfer_fee,
                stamp_duty,
            ) = calculate_explicit_costs(
                trade_value=execution_value,
                side="BUY",
                date=execution_date,
                apply_costs=apply_costs,
            )

            total_cash_required = (
                execution_value
                + commission
                + transfer_fee
            )

            if (
                total_cash_required
                <= cash + 1e-8
            ):
                break

            filled_shares -= BOARD_LOT

        if filled_shares <= 0:
            trade_rows.append(
                make_trade_record(
                    snapshot_date=snapshot_date,
                    execution_date=execution_date,
                    code=code,
                    target_record=target_record,
                    side="BUY",
                    status="SKIPPED",
                    status_reason=(
                        "INSUFFICIENT_CASH"
                    ),
                    requested_value=(
                        requested_value
                    ),
                    filled_mid_value=0.0,
                    filled_shares=0,
                    market_open_price=open_price,
                    execution_price=None,
                    commission=0.0,
                    transfer_fee=0.0,
                    stamp_duty=0.0,
                    slippage_cost=0.0,
                    cash_after=cash,
                )
            )
            continue

        filled_mid_value = (
            filled_shares * open_price
        )

        execution_value = (
            filled_shares
            * execution_price
        )

        slippage_cost = (
            execution_value
            - filled_mid_value
        )

        cash -= (
            execution_value
            + commission
            + transfer_fee
        )

        if cash < -1e-6:
            raise RuntimeError(
                "Cash balance became negative."
            )

        cash = max(cash, 0.0)

        positions[code] = (
            positions.get(code, 0.0)
            + filled_mid_value
        )

        executed_buys += 1

        trade_rows.append(
            make_trade_record(
                snapshot_date=snapshot_date,
                execution_date=execution_date,
                code=code,
                target_record=target_record,
                side="BUY",
                status="EXECUTED",
                status_reason="FILLED_AT_OPEN",
                requested_value=(
                    requested_value
                ),
                filled_mid_value=(
                    filled_mid_value
                ),
                filled_shares=(
                    filled_shares
                ),
                market_open_price=open_price,
                execution_price=(
                    execution_price
                ),
                commission=commission,
                transfer_fee=transfer_fee,
                stamp_duty=stamp_duty,
                slippage_cost=slippage_cost,
                cash_after=cash,
            )
        )

    total_cost = sum(
        row["total_cost"]
        for row in trade_rows
    )

    audit_row = {
        "snapshot_date": snapshot_date,
        "signal_price_date": snapshot[
            "signal_price_date"
        ].max(),
        "execution_date": execution_date,
        "status": "EXECUTED",
        "target_holdings": len(snapshot),
        "actual_holdings_after_open": (
            len(positions)
        ),
        "portfolio_value_before_trade": (
            portfolio_value_before
        ),
        "cash_after_trade": cash,
        "executed_sell_orders": (
            executed_sells
        ),
        "executed_buy_orders": (
            executed_buys
        ),
        "blocked_orders": blocked_orders,
        "total_transaction_cost": (
            total_cost
        ),
    }

    return (
        positions,
        cash,
        trade_rows,
        audit_row,
    )


def update_benchmark(
    current_value: float,
    date: pd.Timestamp,
    first_date: pd.Timestamp,
    bar_lookup: dict[pd.Timestamp, pd.DataFrame],
) -> float:
    bar = get_bar(
        bar_lookup,
        date,
        BENCHMARK_CODE,
    )

    if bar is None:
        return current_value

    if date == first_date:
        if (
            pd.notna(bar["open"])
            and pd.notna(bar["close"])
            and bar["open"] > 0
        ):
            return (
                current_value
                * float(bar["close"])
                / float(bar["open"])
            )

        return current_value

    if pd.notna(bar["pct_change"]):
        return (
            current_value
            * (
                1.0
                + float(bar["pct_change"])
                / 100.0
            )
        )

    if (
        pd.notna(bar["close"])
        and pd.notna(bar["pre_close"])
        and bar["pre_close"] > 0
    ):
        return (
            current_value
            * float(bar["close"])
            / float(bar["pre_close"])
        )

    return current_value


def simulate_backtest(
    targets: pd.DataFrame,
    prices: pd.DataFrame,
    apply_costs: bool,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    trading_calendar = (
        build_trading_calendar(prices)
    )

    bar_lookup = build_bar_lookup(
        prices
    )

    schedule, unscheduled_rows = (
        build_execution_schedule(
            targets=targets,
            trading_calendar=(
                trading_calendar
            ),
        )
    )

    if not schedule:
        raise RuntimeError(
            "No target portfolio has a future execution date."
        )

    first_execution_date = min(
        schedule
    )

    simulation_dates = [
        date
        for date in trading_calendar
        if date >= first_execution_date
    ]

    cash = INITIAL_CAPITAL
    positions: dict[str, float] = {}

    benchmark_value = INITIAL_CAPITAL

    daily_rows = []
    trade_rows = []
    audit_rows = list(
        unscheduled_rows
    )
    position_rows = []

    for date in simulation_dates:
        update_positions_to_open(
            positions=positions,
            date=date,
            bar_lookup=bar_lookup,
        )

        is_rebalance_date = (
            date in schedule
        )

        if is_rebalance_date:
            (
                positions,
                cash,
                new_trade_rows,
                audit_row,
            ) = execute_rebalance(
                snapshot=schedule[date],
                execution_date=date,
                positions=positions,
                cash=cash,
                bar_lookup=bar_lookup,
                apply_costs=apply_costs,
            )

            trade_rows.extend(
                new_trade_rows
            )
            audit_rows.append(
                audit_row
            )

        update_positions_to_close(
            positions=positions,
            date=date,
            bar_lookup=bar_lookup,
        )

        invested_value = float(
            sum(positions.values())
        )

        portfolio_value = (
            cash + invested_value
        )

        if portfolio_value <= 0:
            raise RuntimeError(
                "Portfolio value became non-positive."
            )

        benchmark_value = update_benchmark(
            current_value=benchmark_value,
            date=date,
            first_date=(
                first_execution_date
            ),
            bar_lookup=bar_lookup,
        )

        daily_rows.append(
            {
                "date": date,
                "cash": cash,
                "invested_value": (
                    invested_value
                ),
                "portfolio_value": (
                    portfolio_value
                ),
                "benchmark_value": (
                    benchmark_value
                ),
                "number_of_holdings": (
                    len(positions)
                ),
                "is_rebalance_date": int(
                    is_rebalance_date
                ),
            }
        )

        if is_rebalance_date:
            snapshot_date = schedule[
                date
            ]["snapshot_date"].iloc[0]

            for code, position_value in (
                positions.items()
            ):
                bar = get_bar(
                    bar_lookup,
                    date,
                    code,
                )

                close_price = (
                    float(bar["close"])
                    if (
                        bar is not None
                        and pd.notna(
                            bar["close"]
                        )
                        and bar["close"] > 0
                    )
                    else np.nan
                )

                equivalent_shares = (
                    position_value
                    / close_price
                    if (
                        pd.notna(close_price)
                        and close_price > 0
                    )
                    else np.nan
                )

                position_rows.append(
                    {
                        "snapshot_date": (
                            snapshot_date
                        ),
                        "execution_date": date,
                        "baostock_code": code,
                        "position_value": (
                            position_value
                        ),
                        "equivalent_shares": (
                            equivalent_shares
                        ),
                        "close_price": (
                            close_price
                        ),
                        "portfolio_weight": (
                            position_value
                            / portfolio_value
                        ),
                    }
                )

    daily = pd.DataFrame(daily_rows)

    daily["daily_return"] = daily[
        "portfolio_value"
    ].pct_change(fill_method=None)

    daily["benchmark_daily_return"] = daily[
        "benchmark_value"
    ].pct_change(fill_method=None)

    trades = pd.DataFrame(
        trade_rows
    )

    audit = pd.DataFrame(
        audit_rows
    )

    position_history = pd.DataFrame(
        position_rows
    )

    return (
        daily,
        trades,
        audit,
        position_history,
    )


def calculate_performance(
    values: pd.Series,
    dates: pd.Series,
    method: str,
    total_cost: float,
) -> dict:
    values = pd.to_numeric(
        values,
        errors="coerce",
    )

    valid = pd.DataFrame(
        {
            "date": dates,
            "value": values,
        }
    ).dropna()

    first_value = INITIAL_CAPITAL
    final_value = float(
        valid["value"].iloc[-1]
    )

    total_return = (
        final_value / first_value - 1.0
    )

    calendar_days = max(
        1,
        (
            valid["date"].iloc[-1]
            - valid["date"].iloc[0]
        ).days,
    )

    years = calendar_days / 365.25

    annual_return = (
        (final_value / first_value)
        ** (1.0 / years)
        - 1.0
        if years > 0
        else np.nan
    )

    daily_returns = valid[
        "value"
    ].pct_change(
        fill_method=None
    ).dropna()

    annual_volatility = (
        daily_returns.std(ddof=1)
        * math.sqrt(252)
    )

    if (
        pd.notna(annual_volatility)
        and annual_volatility > 1e-12
    ):
        sharpe_ratio = (
            annual_return
            - RISK_FREE_RATE
        ) / annual_volatility
    else:
        sharpe_ratio = np.nan

    running_peak = valid[
        "value"
    ].cummax()

    drawdown = (
        valid["value"]
        / running_peak
        - 1.0
    )

    maximum_drawdown = (
        drawdown.min()
    )

    return {
        "method": method,
        "start_date": (
            valid["date"].iloc[0]
        ),
        "end_date": (
            valid["date"].iloc[-1]
        ),
        "number_of_days": len(valid),
        "initial_value": first_value,
        "final_value": final_value,
        "total_return_percent": (
            total_return * 100
        ),
        "annual_return_percent": (
            annual_return * 100
        ),
        "annual_volatility_percent": (
            annual_volatility * 100
        ),
        "sharpe_ratio": sharpe_ratio,
        "maximum_drawdown_percent": (
            maximum_drawdown * 100
        ),
        "total_transaction_cost": (
            total_cost
        ),
        "total_cost_percent_of_initial": (
            total_cost
            / INITIAL_CAPITAL
            * 100
        ),
    }


def build_performance_summary(
    daily: pd.DataFrame,
    trades: pd.DataFrame,
) -> pd.DataFrame:
    if trades.empty:
        total_cost = 0.0
    else:
        executed = trades.loc[
            trades["status"]
            == "EXECUTED"
        ]

        total_cost = float(
            executed["total_cost"].sum()
        )

    rows = [
        calculate_performance(
            values=daily[
                "portfolio_value"
            ],
            dates=daily["date"],
            method=(
                "Walk-Forward Portfolio "
                "After Costs"
            ),
            total_cost=total_cost,
        ),
        calculate_performance(
            values=daily[
                "no_cost_portfolio_value"
            ],
            dates=daily["date"],
            method=(
                "Walk-Forward Portfolio "
                "Before Costs"
            ),
            total_cost=0.0,
        ),
        calculate_performance(
            values=daily[
                "benchmark_value"
            ],
            dates=daily["date"],
            method="CSI 300 Benchmark",
            total_cost=0.0,
        ),
    ]

    return pd.DataFrame(rows)


def format_dates(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    output = frame.copy()

    for column in output.columns:
        if (
            column == "date"
            or column.endswith("_date")
        ):
            output[column] = pd.to_datetime(
                output[column],
                errors="coerce",
            ).dt.strftime("%Y-%m-%d")

    return output


def save_outputs(
    connection: sqlite3.Connection,
    daily: pd.DataFrame,
    trades: pd.DataFrame,
    positions: pd.DataFrame,
    audit: pd.DataFrame,
    performance: pd.DataFrame,
) -> None:
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_items = [
        (
            DAILY_TABLE,
            format_dates(daily),
            DAILY_FILE,
        ),
        (
            TRADE_TABLE,
            format_dates(trades),
            TRADE_FILE,
        ),
        (
            POSITION_TABLE,
            format_dates(positions),
            POSITION_FILE,
        ),
        (
            AUDIT_TABLE,
            format_dates(audit),
            AUDIT_FILE,
        ),
        (
            PERFORMANCE_TABLE,
            format_dates(performance),
            PERFORMANCE_FILE,
        ),
    ]

    for (
        table_name,
        frame,
        csv_path,
    ) in output_items:
        frame.to_sql(
            table_name,
            connection,
            if_exists="replace",
            index=False,
            chunksize=2_000,
        )

        frame.to_csv(
            csv_path,
            index=False,
            encoding="utf-8-sig",
            float_format="%.10f",
        )

    connection.execute(
        f"""
        CREATE UNIQUE INDEX IF NOT EXISTS
        idx_{DAILY_TABLE}_date
        ON {DAILY_TABLE} (date);
        """
    )

    connection.commit()


def main() -> None:
    pd.set_option(
        "display.max_columns",
        None,
    )
    pd.set_option(
        "display.width",
        220,
    )

    database_path = resolve_database_path()

    print("=" * 130)
    print("V2 EVENT-DRIVEN BACKTEST")
    print("=" * 130)
    print(f"Database: {database_path}")
    print(
        f"Initial capital: "
        f"¥{INITIAL_CAPITAL:,.2f}"
    )
    print(
        f"Commission: "
        f"{COMMISSION_RATE:.3%}, "
        f"minimum ¥{MINIMUM_COMMISSION:.2f}"
    )
    print(
        f"Slippage per side: "
        f"{SLIPPAGE_RATE:.3%}"
    )
    print(
        f"Board lot: {BOARD_LOT} shares"
    )
    print()

    with sqlite3.connect(
        database_path
    ) as connection:
        for table_name in [
            TARGET_TABLE,
            PRICE_TABLE,
        ]:
            if not table_exists(
                connection,
                table_name,
            ):
                raise RuntimeError(
                    "Required table does not exist: "
                    f"{table_name}"
                )

        print(
            "Loading target portfolios...",
            flush=True,
        )
        targets = load_targets(connection)

        print(
            f"Target snapshots: "
            f"{targets['snapshot_date'].nunique():,}"
        )

        print(
            "Loading unadjusted execution prices...",
            flush=True,
        )
        prices = load_execution_prices(
            connection
        )

        print(
            f"Execution-price rows: "
            f"{len(prices):,}"
        )

        print(
            "Running realistic-cost backtest...",
            flush=True,
        )

        (
            realistic_daily,
            realistic_trades,
            realistic_audit,
            realistic_positions,
        ) = simulate_backtest(
            targets=targets,
            prices=prices,
            apply_costs=True,
        )

        print(
            "Running zero-cost comparison...",
            flush=True,
        )

        (
            no_cost_daily,
            _,
            _,
            _,
        ) = simulate_backtest(
            targets=targets,
            prices=prices,
            apply_costs=False,
        )

        realistic_daily = (
            realistic_daily.merge(
                no_cost_daily[
                    [
                        "date",
                        "portfolio_value",
                    ]
                ].rename(
                    columns={
                        "portfolio_value": (
                            "no_cost_portfolio_value"
                        )
                    }
                ),
                on="date",
                how="left",
                validate="one_to_one",
            )
        )

        realistic_daily[
            "cost_drag_value"
        ] = (
            realistic_daily[
                "no_cost_portfolio_value"
            ]
            - realistic_daily[
                "portfolio_value"
            ]
        )

        performance = (
            build_performance_summary(
                daily=realistic_daily,
                trades=realistic_trades,
            )
        )

        if realistic_trades.empty:
            executed_trades = 0
            blocked_trades = 0
            skipped_trades = 0
        else:
            executed_trades = int(
                (
                    realistic_trades["status"]
                    == "EXECUTED"
                ).sum()
            )

            blocked_trades = int(
                (
                    realistic_trades["status"]
                    == "BLOCKED"
                ).sum()
            )

            skipped_trades = int(
                (
                    realistic_trades["status"]
                    == "SKIPPED"
                ).sum()
            )

        negative_cash_days = int(
            (
                realistic_daily["cash"] < -1e-6
            ).sum()
        )

        print()
        print("=" * 130)
        print("EXECUTION AUDIT")
        print("=" * 130)
        print(
            f"Backtest days: "
            f"{len(realistic_daily):,}"
        )
        print(
            f"Executed trades: "
            f"{executed_trades:,}"
        )
        print(
            f"Blocked trades: "
            f"{blocked_trades:,}"
        )
        print(
            f"Skipped small/cash-limited trades: "
            f"{skipped_trades:,}"
        )
        print(
            f"Negative-cash days: "
            f"{negative_cash_days:,}"
        )
        print(
            f"Final actual holdings: "
            f"{int(realistic_daily['number_of_holdings'].iloc[-1])}"
        )

        if negative_cash_days:
            raise RuntimeError(
                "Negative cash audit failed."
            )

        print(
            "Saving event-backtest results...",
            flush=True,
        )

        save_outputs(
            connection=connection,
            daily=realistic_daily,
            trades=realistic_trades,
            positions=realistic_positions,
            audit=realistic_audit,
            performance=performance,
        )

    print()
    print("=" * 130)
    print("V2 EVENT-BACKTEST PERFORMANCE")
    print("=" * 130)

    display_columns = [
        "method",
        "start_date",
        "end_date",
        "final_value",
        "total_return_percent",
        "annual_return_percent",
        "annual_volatility_percent",
        "sharpe_ratio",
        "maximum_drawdown_percent",
        "total_cost_percent_of_initial",
    ]

    print(
        performance[
            display_columns
        ].to_string(
            index=False,
            formatters={
                "final_value": (
                    lambda value: f"¥{value:,.2f}"
                ),
                "total_return_percent": (
                    lambda value: f"{value:.2f}%"
                ),
                "annual_return_percent": (
                    lambda value: f"{value:.2f}%"
                ),
                "annual_volatility_percent": (
                    lambda value: f"{value:.2f}%"
                ),
                "sharpe_ratio": (
                    lambda value: f"{value:.3f}"
                ),
                "maximum_drawdown_percent": (
                    lambda value: f"{value:.2f}%"
                ),
                "total_cost_percent_of_initial": (
                    lambda value: f"{value:.2f}%"
                ),
            },
        )
    )

    print()
    print("=" * 130)
    print("OUTPUT FILES")
    print("=" * 130)
    print(DAILY_FILE)
    print(TRADE_FILE)
    print(POSITION_FILE)
    print(AUDIT_FILE)
    print(PERFORMANCE_FILE)
    print()
    print(
        "V2 event-driven backtest completed."
    )


if __name__ == "__main__":
    main()