from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "reports" / "results"
DATABASE_PATH = PROJECT_ROOT / "data" / "quant_data.db"

DAILY_FILE = RESULTS_DIR / "event_backtest_daily_v2.csv"
TRADES_FILE = RESULTS_DIR / "event_backtest_trades_v2.csv"

INITIAL_CAPITAL = 1_000_000.0
TRADING_DAYS = 252
RISK_FREE_RATE = 0.02

METHOD_COLUMNS = {
    "Walk-Forward Portfolio After Costs": [
        "portfolio_value",
        "portfolio_value_after_costs",
        "after_cost_portfolio_value",
        "portfolio_after_costs",
        "realistic_portfolio_value",
    ],
    "Walk-Forward Portfolio Before Costs": [
        "no_cost_portfolio_value",
        "portfolio_value_before_costs",
        "before_cost_portfolio_value",
        "portfolio_before_costs",
        "zero_cost_portfolio_value",
    ],
    "CSI 300 Benchmark": [
        "benchmark_value",
        "csi300_value",
        "benchmark_portfolio_value",
    ],
}

def resolve_column(
    dataframe: pd.DataFrame,
    candidates: list[str],
    required: bool = True,
) -> str | None:
    column_lookup = {
        str(column).strip().lower(): column
        for column in dataframe.columns
    }

    for candidate in candidates:
        if candidate.lower() in column_lookup:
            return column_lookup[candidate.lower()]

    if required:
        raise KeyError(
            f"Could not find any of these columns: {candidates}\n"
            f"Available columns: {list(dataframe.columns)}"
        )

    return None


def load_daily_data() -> tuple[pd.DataFrame, str, dict[str, str]]:
    if not DAILY_FILE.exists():
        raise FileNotFoundError(f"Daily backtest file not found: {DAILY_FILE}")

    daily_data = pd.read_csv(DAILY_FILE)

    date_column = resolve_column(
        daily_data,
        ["date", "trading_date", "trade_date"],
    )

    daily_data[date_column] = pd.to_datetime(
        daily_data[date_column],
        errors="coerce",
    )
    daily_data = daily_data.dropna(subset=[date_column])
    daily_data = daily_data.sort_values(date_column)
    daily_data = daily_data.drop_duplicates(
        subset=[date_column],
        keep="last",
    )

    resolved_methods: dict[str, str] = {}

    for method, candidates in METHOD_COLUMNS.items():
        resolved_methods[method] = resolve_column(
            daily_data,
            candidates,
        )

    return daily_data, date_column, resolved_methods


def build_nav_series(
    daily_data: pd.DataFrame,
    date_column: str,
    value_column: str,
) -> pd.Series:
    values = pd.to_numeric(
        daily_data[value_column],
        errors="coerce",
    )

    series = pd.Series(
        values.to_numpy(),
        index=pd.DatetimeIndex(daily_data[date_column]),
        name=value_column,
    )

    series = series.dropna()
    series = series[series > 0]
    series = series[~series.index.duplicated(keep="last")]
    return series.sort_index()


def calculate_daily_returns(nav: pd.Series) -> pd.Series:
    previous_values = nav.shift(1)

    if len(previous_values) > 0:
        previous_values.iloc[0] = INITIAL_CAPITAL

    returns = nav / previous_values - 1.0
    return returns.replace([np.inf, -np.inf], np.nan).dropna()


def maximum_loss_streak(returns: pd.Series) -> int:
    longest_streak = 0
    current_streak = 0

    for value in returns:
        if value < 0:
            current_streak += 1
            longest_streak = max(longest_streak, current_streak)
        else:
            current_streak = 0

    return longest_streak


def calculate_drawdown_details(
    method: str,
    nav: pd.Series,
) -> dict:
    running_maximum = np.maximum.accumulate(
        np.concatenate([[INITIAL_CAPITAL], nav.to_numpy()])
    )[1:]

    drawdowns = nav.to_numpy() / running_maximum - 1.0
    trough_position = int(np.argmin(drawdowns))
    trough_date = nav.index[trough_position]
    trough_value = float(nav.iloc[trough_position])
    peak_value = float(running_maximum[trough_position])

    previous_values = nav.iloc[: trough_position + 1]
    peak_matches = np.where(
        np.isclose(
            previous_values.to_numpy(),
            peak_value,
            rtol=1e-10,
            atol=1e-8,
        )
    )[0]

    if len(peak_matches) > 0:
        peak_position = int(peak_matches[-1])
        peak_date = nav.index[peak_position]
    else:
        peak_position = -1
        peak_date = nav.index[0]

    recovery_date = pd.NaT
    values_after_trough = nav.iloc[trough_position + 1 :]

    recovered = values_after_trough[
        values_after_trough >= peak_value
    ]

    if not recovered.empty:
        recovery_date = recovered.index[0]

    drawdown_trading_days = trough_position - peak_position

    if pd.isna(recovery_date):
        recovery_trading_days = np.nan
        recovery_calendar_days = np.nan
    else:
        recovery_position = nav.index.get_loc(recovery_date)
        recovery_trading_days = recovery_position - trough_position
        recovery_calendar_days = (
            recovery_date - trough_date
        ).days

    return {
        "method": method,
        "maximum_drawdown": float(drawdowns[trough_position]),
        "maximum_drawdown_percent": float(
            drawdowns[trough_position] * 100
        ),
        "peak_date": peak_date,
        "peak_value": peak_value,
        "trough_date": trough_date,
        "trough_value": trough_value,
        "recovery_date": recovery_date,
        "drawdown_trading_days": drawdown_trading_days,
        "drawdown_calendar_days": (
            trough_date - peak_date
        ).days,
        "recovery_trading_days": recovery_trading_days,
        "recovery_calendar_days": recovery_calendar_days,
    }


def calculate_performance(
    method: str,
    nav: pd.Series,
) -> dict:
    returns = calculate_daily_returns(nav)
    number_of_days = len(returns)

    total_return = nav.iloc[-1] / INITIAL_CAPITAL - 1.0
    annual_return = (
        (nav.iloc[-1] / INITIAL_CAPITAL)
        ** (TRADING_DAYS / number_of_days)
        - 1.0
    )

    annual_volatility = (
        returns.std(ddof=1) * np.sqrt(TRADING_DAYS)
    )

    risk_free_daily = (
        (1.0 + RISK_FREE_RATE) ** (1.0 / TRADING_DAYS) - 1.0
    )
    excess_returns = returns - risk_free_daily

    if returns.std(ddof=1) > 0:
        sharpe_ratio = (
            excess_returns.mean()
            / returns.std(ddof=1)
            * np.sqrt(TRADING_DAYS)
        )
    else:
        sharpe_ratio = np.nan

    downside_deviation = np.sqrt(
        np.mean(np.minimum(excess_returns, 0.0) ** 2)
    ) * np.sqrt(TRADING_DAYS)

    if downside_deviation > 0:
        sortino_ratio = (
            excess_returns.mean()
            * TRADING_DAYS
            / downside_deviation
        )
    else:
        sortino_ratio = np.nan

    drawdown_details = calculate_drawdown_details(method, nav)
    maximum_drawdown = drawdown_details["maximum_drawdown"]

    if maximum_drawdown < 0:
        calmar_ratio = annual_return / abs(maximum_drawdown)
    else:
        calmar_ratio = np.nan

    return_threshold = returns.quantile(0.05)
    tail_returns = returns[returns <= return_threshold]

    var_95 = max(0.0, -return_threshold)
    cvar_95 = (
        max(0.0, -tail_returns.mean())
        if not tail_returns.empty
        else np.nan
    )

    best_day = returns.idxmax()
    worst_day = returns.idxmin()

    return {
        "method": method,
        "start_date": nav.index[0],
        "end_date": nav.index[-1],
        "number_of_days": number_of_days,
        "initial_value": INITIAL_CAPITAL,
        "final_value": float(nav.iloc[-1]),
        "total_return_percent": total_return * 100,
        "annual_return_percent": annual_return * 100,
        "annual_volatility_percent": annual_volatility * 100,
        "downside_deviation_percent": downside_deviation * 100,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "maximum_drawdown_percent": maximum_drawdown * 100,
        "calmar_ratio": calmar_ratio,
        "var_95_percent": var_95 * 100,
        "cvar_95_percent": cvar_95 * 100,
        "daily_win_rate_percent": (returns > 0).mean() * 100,
        "daily_loss_rate_percent": (returns < 0).mean() * 100,
        "maximum_loss_streak_days": maximum_loss_streak(returns),
        "best_day": best_day,
        "best_day_return_percent": returns.loc[best_day] * 100,
        "worst_day": worst_day,
        "worst_day_return_percent": returns.loc[worst_day] * 100,
    }


def calculate_relative_performance(
    portfolio_nav: pd.Series,
    benchmark_nav: pd.Series,
) -> pd.DataFrame:
    portfolio_returns = calculate_daily_returns(portfolio_nav)
    benchmark_returns = calculate_daily_returns(benchmark_nav)

    aligned = pd.concat(
        [
            portfolio_returns.rename("portfolio_return"),
            benchmark_returns.rename("benchmark_return"),
        ],
        axis=1,
        join="inner",
    ).dropna()

    active_returns = (
        aligned["portfolio_return"]
        - aligned["benchmark_return"]
    )

    tracking_error = (
        active_returns.std(ddof=1) * np.sqrt(TRADING_DAYS)
    )

    if tracking_error > 0:
        information_ratio = (
            active_returns.mean()
            / active_returns.std(ddof=1)
            * np.sqrt(TRADING_DAYS)
        )
    else:
        information_ratio = np.nan

    benchmark_variance = aligned["benchmark_return"].var(ddof=1)

    if benchmark_variance > 0:
        beta = (
            aligned["portfolio_return"].cov(
                aligned["benchmark_return"]
            )
            / benchmark_variance
        )
    else:
        beta = np.nan

    risk_free_daily = (
        (1.0 + RISK_FREE_RATE) ** (1.0 / TRADING_DAYS) - 1.0
    )

    alpha_daily = (
        aligned["portfolio_return"].mean()
        - risk_free_daily
        - beta
        * (
            aligned["benchmark_return"].mean()
            - risk_free_daily
        )
    )
    annual_alpha = alpha_daily * TRADING_DAYS

    correlation = aligned["portfolio_return"].corr(
        aligned["benchmark_return"]
    )

    positive_benchmark = aligned[
        aligned["benchmark_return"] > 0
    ]
    negative_benchmark = aligned[
        aligned["benchmark_return"] < 0
    ]

    up_capture = np.nan
    down_capture = np.nan

    if (
        not positive_benchmark.empty
        and positive_benchmark["benchmark_return"].mean() != 0
    ):
        up_capture = (
            positive_benchmark["portfolio_return"].mean()
            / positive_benchmark["benchmark_return"].mean()
            * 100
        )

    if (
        not negative_benchmark.empty
        and negative_benchmark["benchmark_return"].mean() != 0
    ):
        down_capture = (
            negative_benchmark["portfolio_return"].mean()
            / negative_benchmark["benchmark_return"].mean()
            * 100
        )

    portfolio_total_return = (
        portfolio_nav.iloc[-1] / INITIAL_CAPITAL - 1.0
    )
    benchmark_total_return = (
        benchmark_nav.iloc[-1] / INITIAL_CAPITAL - 1.0
    )

    return pd.DataFrame(
        [
            {
                "comparison": (
                    "Walk-Forward Portfolio After Costs "
                    "vs CSI 300"
                ),
                "number_of_days": len(aligned),
                "portfolio_total_return_percent": (
                    portfolio_total_return * 100
                ),
                "benchmark_total_return_percent": (
                    benchmark_total_return * 100
                ),
                "cumulative_excess_return_percent": (
                    portfolio_total_return
                    - benchmark_total_return
                )
                * 100,
                "annual_tracking_error_percent": (
                    tracking_error * 100
                ),
                "information_ratio": information_ratio,
                "beta": beta,
                "annual_alpha_percent": annual_alpha * 100,
                "return_correlation": correlation,
                "up_capture_percent": up_capture,
                "down_capture_percent": down_capture,
            }
        ]
    )


def calculate_period_returns(
    method: str,
    nav: pd.Series,
    frequency: str,
    period_type: str,
) -> pd.DataFrame:
    data = pd.DataFrame(
        {
            "date": nav.index,
            "portfolio_value": nav.to_numpy(),
        }
    )

    if frequency == "year":
        data["period"] = data["date"].dt.to_period("Y").astype(str)
    else:
        data["period"] = data["date"].dt.to_period("M").astype(str)

    result_rows = []
    previous_value = INITIAL_CAPITAL

    for period, group in data.groupby("period", sort=True):
        group = group.sort_values("date")
        ending_value = float(group["portfolio_value"].iloc[-1])

        period_return = ending_value / previous_value - 1.0

        result_rows.append(
            {
                "method": method,
                "period_type": period_type,
                "period": period,
                "period_start_date": group["date"].iloc[0],
                "period_end_date": group["date"].iloc[-1],
                "starting_value": previous_value,
                "ending_value": ending_value,
                "return_percent": period_return * 100,
            }
        )

        previous_value = ending_value

    return pd.DataFrame(result_rows)


def rolling_maximum_drawdown(values: np.ndarray) -> float:
    running_maximum = np.maximum.accumulate(values)
    drawdowns = values / running_maximum - 1.0
    return float(np.min(drawdowns))


def calculate_rolling_metrics(
    method: str,
    nav: pd.Series,
) -> pd.DataFrame:
    returns = calculate_daily_returns(nav)

    rolling_return = (
        (1.0 + returns)
        .rolling(TRADING_DAYS)
        .apply(np.prod, raw=True)
        - 1.0
    )

    rolling_volatility = (
        returns.rolling(TRADING_DAYS).std(ddof=1)
        * np.sqrt(TRADING_DAYS)
    )

    risk_free_daily = (
        (1.0 + RISK_FREE_RATE) ** (1.0 / TRADING_DAYS) - 1.0
    )

    rolling_sharpe = (
        (returns - risk_free_daily)
        .rolling(TRADING_DAYS)
        .mean()
        / returns.rolling(TRADING_DAYS).std(ddof=1)
        * np.sqrt(TRADING_DAYS)
    )

    rolling_drawdown = nav.rolling(TRADING_DAYS).apply(
        rolling_maximum_drawdown,
        raw=True,
    )

    result = pd.DataFrame(
        {
            "date": nav.index,
            "method": method,
            "rolling_12m_return_percent": (
                rolling_return.reindex(nav.index) * 100
            ),
            "rolling_12m_volatility_percent": (
                rolling_volatility.reindex(nav.index) * 100
            ),
            "rolling_12m_sharpe_ratio": (
                rolling_sharpe.reindex(nav.index)
            ),
            "rolling_12m_maximum_drawdown_percent": (
                rolling_drawdown * 100
            ),
        }
    )

    return result.dropna(
        subset=[
            "rolling_12m_return_percent",
            "rolling_12m_volatility_percent",
        ]
    )


def calculate_turnover_and_costs(
    after_cost_nav: pd.Series,
) -> pd.DataFrame:
    if not TRADES_FILE.exists():
        return pd.DataFrame(
            [
                {
                    "number_of_executed_trades": 0,
                    "notes": "Trade file not found.",
                }
            ]
        )

    trades = pd.read_csv(TRADES_FILE)

    date_column = resolve_column(
        trades,
        ["execution_date", "trade_date", "date"],
    )
    action_column = resolve_column(
        trades,
        ["action", "side", "trade_action"],
    )

    trades[date_column] = pd.to_datetime(
        trades[date_column],
        errors="coerce",
    )
    trades[action_column] = (
        trades[action_column].astype(str).str.upper()
    )

    status_column = resolve_column(
        trades,
        ["execution_status", "trade_status", "status"],
        required=False,
    )

    if status_column is not None:
        status_text = trades[status_column].astype(str).str.upper()
        executed_mask = status_text.str.contains(
            "EXECUTED|FILLED",
            regex=True,
        )
        if executed_mask.any():
            trades = trades[executed_mask].copy()

    value_column = resolve_column(
        trades,
        [
            "filled_mid_value",
            "executed_value",
            "trade_value",
            "gross_trade_value",
            "filled_value",
        ],
        required=False,
    )

    if value_column is not None:
        trades["_trade_value"] = pd.to_numeric(
            trades[value_column],
            errors="coerce",
        ).abs()
    else:
        shares_column = resolve_column(
            trades,
            ["filled_shares", "executed_shares", "shares"],
        )
        price_column = resolve_column(
            trades,
            [
                "execution_price",
                "executed_price",
                "fill_price",
                "open",
            ],
        )

        trades["_trade_value"] = (
            pd.to_numeric(
                trades[shares_column],
                errors="coerce",
            ).abs()
            * pd.to_numeric(
                trades[price_column],
                errors="coerce",
            ).abs()
        )

    cost_candidates = {
        "commission": ["commission", "commission_cost"],
        "stamp_duty": ["stamp_duty", "stamp_tax"],
        "transfer_fee": ["transfer_fee"],
        "slippage_cost": ["slippage_cost", "slippage"],
        "total_cost": [
            "total_cost",
            "transaction_cost",
            "total_transaction_cost",
        ],
    }

    cost_totals: dict[str, float] = {}

    for cost_name, candidates in cost_candidates.items():
        column = resolve_column(
            trades,
            candidates,
            required=False,
        )

        if column is None:
            cost_totals[cost_name] = 0.0
        else:
            cost_totals[cost_name] = float(
                pd.to_numeric(
                    trades[column],
                    errors="coerce",
                )
                .fillna(0.0)
                .sum()
            )

    if cost_totals["total_cost"] == 0:
        cost_totals["total_cost"] = (
            cost_totals["commission"]
            + cost_totals["stamp_duty"]
            + cost_totals["transfer_fee"]
            + cost_totals["slippage_cost"]
        )

    trades = trades.dropna(
        subset=[date_column, "_trade_value"]
    )

    buy_value = trades.loc[
        trades[action_column] == "BUY",
        "_trade_value",
    ].sum()

    sell_value = trades.loc[
        trades[action_column] == "SELL",
        "_trade_value",
    ].sum()

    gross_trade_value = trades["_trade_value"].sum()
    average_portfolio_value = after_cost_nav.mean()

    total_one_way_turnover = (
        gross_trade_value
        / 2.0
        / average_portfolio_value
    )

    trades["_month"] = trades[date_column].dt.to_period("M")

    first_rebalance_date = trades[date_column].min()
    recurring_trades = trades[
        trades[date_column] > first_rebalance_date
    ].copy()

    monthly_turnovers = []

    for _, group in recurring_trades.groupby("_month"):
        month_start = group[date_column].min()
        month_end = group[date_column].max()

        month_nav = after_cost_nav[
            (after_cost_nav.index >= month_start)
            & (after_cost_nav.index <= month_end)
        ]

        if month_nav.empty:
            denominator = average_portfolio_value
        else:
            denominator = month_nav.mean()

        monthly_turnovers.append(
            group["_trade_value"].sum()
            / 2.0
            / denominator
        )

    average_monthly_turnover = (
        float(np.mean(monthly_turnovers))
        if monthly_turnovers
        else np.nan
    )

    return pd.DataFrame(
        [
            {
                "number_of_executed_trades": len(trades),
                "number_of_rebalances": trades[date_column].nunique(),
                "number_of_buy_trades": int(
                    (trades[action_column] == "BUY").sum()
                ),
                "number_of_sell_trades": int(
                    (trades[action_column] == "SELL").sum()
                ),
                "gross_buy_value": float(buy_value),
                "gross_sell_value": float(sell_value),
                "gross_trade_value": float(gross_trade_value),
                "average_portfolio_value": float(
                    average_portfolio_value
                ),
                "total_one_way_turnover_percent": (
                    total_one_way_turnover * 100
                ),
                "average_monthly_one_way_turnover_percent": (
                    average_monthly_turnover * 100
                ),
                "annualized_one_way_turnover_percent": (
                    average_monthly_turnover * 12 * 100
                ),
                "commission": cost_totals["commission"],
                "stamp_duty": cost_totals["stamp_duty"],
                "transfer_fee": cost_totals["transfer_fee"],
                "slippage_cost": cost_totals["slippage_cost"],
                "total_transaction_cost": cost_totals["total_cost"],
                "total_cost_percent_of_initial": (
                    cost_totals["total_cost"]
                    / INITIAL_CAPITAL
                    * 100
                ),
            }
        ]
    )


def format_date_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    result = dataframe.copy()

    for column in result.columns:
        if column == "date" or column.endswith("_date"):
            converted = pd.to_datetime(
                result[column],
                errors="coerce",
            )
            result[column] = converted.dt.strftime("%Y-%m-%d")
            result.loc[converted.isna(), column] = None

    return result


def save_results(
    performance: pd.DataFrame,
    relative: pd.DataFrame,
    drawdowns: pd.DataFrame,
    annual_returns: pd.DataFrame,
    monthly_returns: pd.DataFrame,
    rolling_metrics: pd.DataFrame,
    turnover_costs: pd.DataFrame,
) -> dict[str, Path]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    outputs = {
        "performance": (
            RESULTS_DIR
            / "event_risk_performance_summary_v2.csv"
        ),
        "relative": (
            RESULTS_DIR
            / "event_relative_performance_v2.csv"
        ),
        "drawdowns": (
            RESULTS_DIR
            / "event_drawdown_details_v2.csv"
        ),
        "annual_returns": (
            RESULTS_DIR
            / "event_annual_returns_v2.csv"
        ),
        "monthly_returns": (
            RESULTS_DIR
            / "event_monthly_returns_v2.csv"
        ),
        "rolling_metrics": (
            RESULTS_DIR
            / "event_rolling_metrics_v2.csv"
        ),
        "turnover_costs": (
            RESULTS_DIR
            / "event_turnover_cost_summary_v2.csv"
        ),
    }

    datasets = {
        "performance": format_date_columns(performance),
        "relative": format_date_columns(relative),
        "drawdowns": format_date_columns(drawdowns),
        "annual_returns": format_date_columns(annual_returns),
        "monthly_returns": format_date_columns(monthly_returns),
        "rolling_metrics": format_date_columns(rolling_metrics),
        "turnover_costs": format_date_columns(turnover_costs),
    }

    for name, dataframe in datasets.items():
        dataframe.to_csv(
            outputs[name],
            index=False,
            encoding="utf-8-sig",
        )

    table_names = {
        "performance": "event_risk_performance_v2",
        "relative": "event_relative_performance_v2",
        "drawdowns": "event_drawdown_details_v2",
        "annual_returns": "event_annual_returns_v2",
        "monthly_returns": "event_monthly_returns_v2",
        "rolling_metrics": "event_rolling_metrics_v2",
        "turnover_costs": "event_turnover_cost_summary_v2",
    }

    with sqlite3.connect(DATABASE_PATH) as connection:
        for name, dataframe in datasets.items():
            dataframe.to_sql(
                table_names[name],
                connection,
                if_exists="replace",
                index=False,
            )

    return outputs


def print_performance_table(performance: pd.DataFrame) -> None:
    display = performance[
        [
            "method",
            "final_value",
            "total_return_percent",
            "annual_return_percent",
            "annual_volatility_percent",
            "sharpe_ratio",
            "sortino_ratio",
            "maximum_drawdown_percent",
            "calmar_ratio",
            "var_95_percent",
            "cvar_95_percent",
        ]
    ].copy()

    display["final_value"] = display["final_value"].map(
        lambda value: f"¥{value:,.2f}"
    )

    percent_columns = [
        "total_return_percent",
        "annual_return_percent",
        "annual_volatility_percent",
        "maximum_drawdown_percent",
        "var_95_percent",
        "cvar_95_percent",
    ]

    for column in percent_columns:
        display[column] = display[column].map(
            lambda value: f"{value:.2f}%"
        )

    for column in [
        "sharpe_ratio",
        "sortino_ratio",
        "calmar_ratio",
    ]:
        display[column] = display[column].map(
            lambda value: (
                f"{value:.3f}"
                if pd.notna(value)
                else "N/A"
            )
        )

    print(display.to_string(index=False))


def main() -> None:
    print("=" * 130)
    print("V2 RISK AND PERFORMANCE DIAGNOSTICS")
    print("=" * 130)
    print(f"Database: {DATABASE_PATH}")
    print(f"Risk-free rate assumption: {RISK_FREE_RATE:.2%}")
    print()

    daily_data, date_column, method_columns = load_daily_data()

    nav_series = {
        method: build_nav_series(
            daily_data,
            date_column,
            value_column,
        )
        for method, value_column in method_columns.items()
    }

    performance = pd.DataFrame(
        [
            calculate_performance(method, nav)
            for method, nav in nav_series.items()
        ]
    )

    drawdowns = pd.DataFrame(
        [
            calculate_drawdown_details(method, nav)
            for method, nav in nav_series.items()
        ]
    )

    relative = calculate_relative_performance(
        nav_series["Walk-Forward Portfolio After Costs"],
        nav_series["CSI 300 Benchmark"],
    )

    annual_returns = pd.concat(
        [
            calculate_period_returns(
                method,
                nav,
                frequency="year",
                period_type="Annual",
            )
            for method, nav in nav_series.items()
        ],
        ignore_index=True,
    )

    monthly_returns = pd.concat(
        [
            calculate_period_returns(
                method,
                nav,
                frequency="month",
                period_type="Monthly",
            )
            for method, nav in nav_series.items()
        ],
        ignore_index=True,
    )

    rolling_metrics = pd.concat(
        [
            calculate_rolling_metrics(method, nav)
            for method, nav in nav_series.items()
        ],
        ignore_index=True,
    )

    turnover_costs = calculate_turnover_and_costs(
        nav_series["Walk-Forward Portfolio After Costs"]
    )

    outputs = save_results(
        performance=performance,
        relative=relative,
        drawdowns=drawdowns,
        annual_returns=annual_returns,
        monthly_returns=monthly_returns,
        rolling_metrics=rolling_metrics,
        turnover_costs=turnover_costs,
    )

    print("=" * 130)
    print("RISK AND PERFORMANCE SUMMARY")
    print("=" * 130)
    print_performance_table(performance)
    print()

    print("=" * 130)
    print("RELATIVE PERFORMANCE")
    print("=" * 130)
    print(relative.round(4).to_string(index=False))
    print()

    print("=" * 130)
    print("MAXIMUM DRAWDOWN DETAILS")
    print("=" * 130)
    print(
        format_date_columns(drawdowns)[
            [
                "method",
                "maximum_drawdown_percent",
                "peak_date",
                "trough_date",
                "recovery_date",
                "drawdown_trading_days",
                "recovery_trading_days",
            ]
        ].to_string(index=False)
    )
    print()

    print("=" * 130)
    print("TURNOVER AND COST SUMMARY")
    print("=" * 130)
    print(turnover_costs.round(4).to_string(index=False))
    print()

    print("=" * 130)
    print("OUTPUT FILES")
    print("=" * 130)

    for output_path in outputs.values():
        print(output_path)

    print()
    print("V2 risk and performance diagnostics completed.")


if __name__ == "__main__":
    main()