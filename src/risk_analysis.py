import sqlite3

import numpy as np
import pandas as pd

from config.settings import (
    DATABASE_PATH,
    RESULT_DIR,
    TRADING_DAYS_PER_YEAR
)


# =====================================
# 1. 读取样本外回测结果
# =====================================

def load_oos_results():

    connection = sqlite3.connect(
        DATABASE_PATH
    )


    query = """
    SELECT
        date,
        trained_return,
        original_return,
        benchmark_return,
        trained_value,
        original_value,
        benchmark_value
    FROM oos_backtest_results
    ORDER BY date;
    """


    results = pd.read_sql_query(
        query,
        connection
    )


    connection.close()


    if results.empty:

        raise ValueError(
            "No out-of-sample results found. "
            "Please run "
            "python -m src.oos_backtest first."
        )


    results["date"] = (
        pd.to_datetime(
            results["date"]
        )
    )


    results = (
        results
        .sort_values("date")
        .set_index("date")
    )


    return results


# =====================================
# 2. 读取样本外调仓指令
# =====================================

def load_oos_orders():

    connection = sqlite3.connect(
        DATABASE_PATH
    )


    query = """
    SELECT
        strategy,
        signal_date,
        symbol,
        target_weight
    FROM oos_testing_orders
    ORDER BY strategy, signal_date, symbol;
    """


    orders = pd.read_sql_query(
        query,
        connection
    )


    connection.close()


    if orders.empty:

        raise ValueError(
            "No out-of-sample orders found."
        )


    orders["signal_date"] = (
        pd.to_datetime(
            orders["signal_date"]
        )
    )


    return orders


# =====================================
# 3. 计算最长连续亏损天数
# =====================================

def calculate_maximum_loss_streak(
    return_series
):

    maximum_streak = 0

    current_streak = 0


    for daily_return in return_series:

        if daily_return < 0:

            current_streak += 1

            maximum_streak = max(
                maximum_streak,
                current_streak
            )

        else:

            current_streak = 0


    return maximum_streak


# =====================================
# 4. 计算月度收益率
# =====================================

def calculate_monthly_returns(
    method_name,
    return_series
):

    monthly_return = (
        (
            1
            + return_series
        )
        .groupby(
            return_series.index.to_period(
                "M"
            )
        )
        .prod()
        - 1
    )


    monthly_data = pd.DataFrame(
        {
            "month": (
                monthly_return.index.astype(
                    str
                )
            ),

            "method": (
                method_name
            ),

            "monthly_return": (
                monthly_return.values
            )
        }
    )


    return monthly_data


# =====================================
# 5. 计算单个方法的风险指标
# =====================================

def calculate_risk_metrics(
    method_name,
    return_series,
    value_series
):

    return_series = (
        return_series
        .dropna()
        .copy()
    )


    value_series = (
        value_series
        .reindex(
            return_series.index
        )
        .dropna()
    )


    # ---------------------------------
    # 年化波动率
    # ---------------------------------

    annual_volatility = (
        return_series.std()
        * np.sqrt(
            TRADING_DAYS_PER_YEAR
        )
    )


    # ---------------------------------
    # 历史模拟95% VaR
    # ---------------------------------

    fifth_percentile = (
        return_series.quantile(
            0.05
        )
    )


    value_at_risk_95 = max(
        0,
        -fifth_percentile
    )


    # ---------------------------------
    # 历史模拟95% CVaR
    # ---------------------------------

    tail_losses = return_series[
        return_series
        <= fifth_percentile
    ]


    if tail_losses.empty:

        conditional_var_95 = 0

    else:

        conditional_var_95 = max(
            0,
            -tail_losses.mean()
        )


    # ---------------------------------
    # 最好和最差交易日
    # ---------------------------------

    best_day = (
        return_series.idxmax()
    )


    worst_day = (
        return_series.idxmin()
    )


    best_day_return = (
        return_series.max()
    )


    worst_day_return = (
        return_series.min()
    )


    # ---------------------------------
    # 最大回撤
    # ---------------------------------

    running_peak = (
        value_series.cummax()
    )


    drawdown = (
        value_series
        / running_peak
        - 1
    )


    maximum_drawdown = (
        drawdown.min()
    )


    # ---------------------------------
    # 连续亏损
    # ---------------------------------

    maximum_loss_streak = (
        calculate_maximum_loss_streak(
            return_series
        )
    )


    # ---------------------------------
    # 日胜率和亏损概率
    # ---------------------------------

    daily_win_rate = (
        (
            return_series > 0
        ).mean()
    )


    daily_loss_rate = (
        (
            return_series < 0
        ).mean()
    )


    # ---------------------------------
    # 月度风险统计
    # ---------------------------------

    monthly_data = (
        calculate_monthly_returns(
            method_name,
            return_series
        )
    )


    monthly_win_rate = (
        (
            monthly_data[
                "monthly_return"
            ] > 0
        ).mean()
    )


    best_month_row = (
        monthly_data.loc[
            monthly_data[
                "monthly_return"
            ].idxmax()
        ]
    )


    worst_month_row = (
        monthly_data.loc[
            monthly_data[
                "monthly_return"
            ].idxmin()
        ]
    )


    risk_metrics = {
        "method": method_name,

        "number_of_days": (
            len(return_series)
        ),

        "annual_volatility_percent": (
            annual_volatility * 100
        ),

        "var_95_percent": (
            value_at_risk_95 * 100
        ),

        "cvar_95_percent": (
            conditional_var_95 * 100
        ),

        "best_day": (
            best_day.strftime(
                "%Y-%m-%d"
            )
        ),

        "best_day_return_percent": (
            best_day_return * 100
        ),

        "worst_day": (
            worst_day.strftime(
                "%Y-%m-%d"
            )
        ),

        "worst_day_return_percent": (
            worst_day_return * 100
        ),

        "maximum_drawdown_percent": (
            maximum_drawdown * 100
        ),

        "maximum_loss_streak_days": (
            maximum_loss_streak
        ),

        "daily_win_rate_percent": (
            daily_win_rate * 100
        ),

        "daily_loss_rate_percent": (
            daily_loss_rate * 100
        ),

        "monthly_win_rate_percent": (
            monthly_win_rate * 100
        ),

        "best_month": (
            best_month_row[
                "month"
            ]
        ),

        "best_month_return_percent": (
            best_month_row[
                "monthly_return"
            ] * 100
        ),

        "worst_month": (
            worst_month_row[
                "month"
            ]
        ),

        "worst_month_return_percent": (
            worst_month_row[
                "monthly_return"
            ] * 100
        )
    }


    return (
        risk_metrics,
        monthly_data
    )


# =====================================
# 6. 计算策略换手率
# =====================================

def calculate_turnover(
    orders
):

    turnover_records = []


    for strategy_name, strategy_orders in (
        orders.groupby("strategy")
    ):

        weight_matrix = (
            strategy_orders
            .pivot_table(
                index="signal_date",
                columns="symbol",
                values="target_weight",
                aggfunc="sum"
            )
            .fillna(0)
            .sort_index()
        )


        weight_changes = (
            weight_matrix
            .diff()
            .abs()
            .sum(axis=1)
            / 2
        )


        # 第一次建仓从现金开始
        # 需要按照100%建仓计算
        if not weight_changes.empty:

            weight_changes.iloc[0] = (
                weight_matrix
                .iloc[0]
                .abs()
                .sum()
            )


        number_of_periods = len(
            weight_changes
        )


        total_turnover = (
            weight_changes.sum()
        )


        average_monthly_turnover = (
            weight_changes.mean()
        )


        annualized_turnover = (
            average_monthly_turnover
            * 12
        )


        turnover_records.append(
            {
                "method": (
                    strategy_name
                ),

                "number_of_rebalances": (
                    number_of_periods
                ),

                "total_turnover_percent": (
                    total_turnover * 100
                ),

                "average_monthly_turnover_percent": (
                    average_monthly_turnover
                    * 100
                ),

                "annualized_turnover_percent": (
                    annualized_turnover
                    * 100
                )
            }
        )


    turnover_summary = pd.DataFrame(
        turnover_records
    )


    benchmark_row = pd.DataFrame(
        [
            {
                "method": (
                    "CSI 300 Benchmark"
                ),

                "number_of_rebalances": 0,

                "total_turnover_percent": 0.0,

                "average_monthly_turnover_percent": (
                    0.0
                ),

                "annualized_turnover_percent": (
                    0.0
                )
            }
        ]
    )


    turnover_summary = pd.concat(
        [
            turnover_summary,
            benchmark_row
        ],
        ignore_index=True
    )


    return turnover_summary


# =====================================
# 7. 保存风险结果
# =====================================

def save_risk_results(
    risk_summary,
    monthly_returns,
    turnover_summary
):

    connection = sqlite3.connect(
        DATABASE_PATH
    )


    risk_summary.to_sql(
        "oos_risk_summary",
        connection,
        if_exists="replace",
        index=False
    )


    monthly_returns.to_sql(
        "oos_monthly_returns",
        connection,
        if_exists="replace",
        index=False
    )


    turnover_summary.to_sql(
        "oos_turnover_summary",
        connection,
        if_exists="replace",
        index=False
    )


    connection.close()


    risk_summary.to_csv(
        RESULT_DIR
        / "oos_risk_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )


    monthly_returns.to_csv(
        RESULT_DIR
        / "oos_monthly_returns.csv",
        index=False,
        encoding="utf-8-sig"
    )


    turnover_summary.to_csv(
        RESULT_DIR
        / "oos_turnover_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )


# =====================================
# 8. 主程序
# =====================================

def main():

    print(
        "\nRunning out-of-sample "
        "risk analysis..."
    )


    results = (
        load_oos_results()
    )


    orders = (
        load_oos_orders()
    )


    method_settings = [
        (
            "Trained Factor",
            "trained_return",
            "trained_value"
        ),

        (
            "Original Factor",
            "original_return",
            "original_value"
        ),

        (
            "CSI 300 Benchmark",
            "benchmark_return",
            "benchmark_value"
        )
    ]


    risk_records = []

    monthly_records = []


    for (
        method_name,
        return_column,
        value_column
    ) in method_settings:

        (
            risk_metrics,
            monthly_data
        ) = calculate_risk_metrics(
            method_name,
            results[
                return_column
            ],
            results[
                value_column
            ]
        )


        risk_records.append(
            risk_metrics
        )


        monthly_records.append(
            monthly_data
        )


    risk_summary = pd.DataFrame(
        risk_records
    )


    monthly_returns = pd.concat(
        monthly_records,
        ignore_index=True
    )


    turnover_summary = (
        calculate_turnover(
            orders
        )
    )


    save_risk_results(
        risk_summary,
        monthly_returns,
        turnover_summary
    )


    pd.set_option(
        "display.max_columns",
        None
    )

    pd.set_option(
        "display.width",
        260
    )

    pd.set_option(
        "display.float_format",
        "{:,.2f}".format
    )


    print("\n" + "=" * 120)

    print("OUT-OF-SAMPLE RISK SUMMARY")

    print("=" * 120)


    print(
        risk_summary.to_string(
            index=False
        )
    )


    print("\n" + "=" * 120)

    print("TURNOVER SUMMARY")

    print("=" * 120)


    print(
        turnover_summary.to_string(
            index=False
        )
    )


    print(
        "\nRisk analysis completed."
    )


# =====================================
# 9. 启动程序
# =====================================

if __name__ == "__main__":

    main()