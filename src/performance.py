import sqlite3

import numpy as np
import pandas as pd

from config.settings import (
    DATABASE_PATH,
    RESULT_DIR,
    INITIAL_CAPITAL,
    TRADING_DAYS_PER_YEAR
)


# =====================================
# 1. 无风险利率假设
# =====================================

ANNUAL_RISK_FREE_RATE = 0.02


# =====================================
# 2. 读取回测结果
# =====================================

def load_backtest_results():

    connection = sqlite3.connect(
        DATABASE_PATH
    )


    query = """
    SELECT
        date,
        portfolio_value,
        benchmark_value,
        daily_return,
        benchmark_return,
        transaction_cost,
        drawdown
    FROM backtest_results
    ORDER BY date;
    """


    results = pd.read_sql_query(
        query,
        connection
    )


    connection.close()


    if results.empty:

        raise ValueError(
            "No backtest results found. "
            "Please run "
            "python -m src.backtester first."
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


    numeric_columns = [
        "portfolio_value",
        "benchmark_value",
        "daily_return",
        "benchmark_return",
        "transaction_cost",
        "drawdown"
    ]


    for column in numeric_columns:

        results[column] = (
            pd.to_numeric(
                results[column],
                errors="coerce"
            )
        )


    results = results.dropna(
        subset=[
            "portfolio_value",
            "benchmark_value",
            "daily_return",
            "benchmark_return"
        ]
    )


    return results


# =====================================
# 3. 计算最大回撤详细日期
# =====================================

def calculate_drawdown_details(
    value_series
):

    running_peak = (
        value_series.cummax()
    )


    drawdown = (
        value_series
        / running_peak
        - 1
    )


    trough_date = (
        drawdown.idxmin()
    )


    maximum_drawdown = (
        drawdown.loc[
            trough_date
        ]
    )


    values_before_trough = (
        value_series.loc[
            :trough_date
        ]
    )


    peak_date = (
        values_before_trough.idxmax()
    )


    peak_value = (
        value_series.loc[
            peak_date
        ]
    )


    values_after_trough = (
        value_series.loc[
            trough_date:
        ]
    )


    recovery_candidates = (
        values_after_trough[
            values_after_trough
            >= peak_value
        ]
    )


    if recovery_candidates.empty:

        recovery_date = None

    else:

        recovery_date = (
            recovery_candidates.index[0]
        )


    drawdown_days = (
        trough_date
        - peak_date
    ).days


    if recovery_date is None:

        recovery_days = None

    else:

        recovery_days = (
            recovery_date
            - trough_date
        ).days


    return {
        "maximum_drawdown": (
            maximum_drawdown
        ),

        "peak_date": (
            peak_date
        ),

        "trough_date": (
            trough_date
        ),

        "recovery_date": (
            recovery_date
        ),

        "drawdown_days": (
            drawdown_days
        ),

        "recovery_days": (
            recovery_days
        )
    }


# =====================================
# 4. 计算单个投资方法的绩效
# =====================================

def calculate_performance_metrics(
    method_name,
    value_series,
    return_series
):

    value_series = (
        value_series
        .dropna()
        .copy()
    )


    return_series = (
        return_series
        .reindex(
            value_series.index
        )
        .fillna(0)
        .copy()
    )


    number_of_days = len(
        return_series
    )


    number_of_years = (
        number_of_days
        / TRADING_DAYS_PER_YEAR
    )


    final_value = (
        value_series.iloc[-1]
    )


    total_return = (
        final_value
        / INITIAL_CAPITAL
        - 1
    )


    if number_of_years > 0:

        annual_return = (
            (
                final_value
                / INITIAL_CAPITAL
            )
            ** (
                1
                / number_of_years
            )
            - 1
        )

    else:

        annual_return = 0


    daily_volatility = (
        return_series.std()
    )


    annual_volatility = (
        daily_volatility
        * np.sqrt(
            TRADING_DAYS_PER_YEAR
        )
    )


    daily_risk_free_rate = (
        (
            1
            + ANNUAL_RISK_FREE_RATE
        )
        ** (
            1
            / TRADING_DAYS_PER_YEAR
        )
        - 1
    )


    excess_daily_return = (
        return_series
        - daily_risk_free_rate
    )


    if daily_volatility == 0:

        sharpe_ratio = 0

    else:

        sharpe_ratio = (
            excess_daily_return.mean()
            / daily_volatility
            * np.sqrt(
                TRADING_DAYS_PER_YEAR
            )
        )


    downside_returns = (
        np.minimum(
            excess_daily_return,
            0
        )
    )


    downside_deviation = (
        np.sqrt(
            np.mean(
                downside_returns ** 2
            )
        )
        * np.sqrt(
            TRADING_DAYS_PER_YEAR
        )
    )


    if downside_deviation == 0:

        sortino_ratio = 0

    else:

        sortino_ratio = (
            annual_return
            - ANNUAL_RISK_FREE_RATE
        ) / downside_deviation


    drawdown_details = (
        calculate_drawdown_details(
            value_series
        )
    )


    maximum_drawdown = (
        drawdown_details[
            "maximum_drawdown"
        ]
    )


    if maximum_drawdown == 0:

        calmar_ratio = 0

    else:

        calmar_ratio = (
            annual_return
            / abs(
                maximum_drawdown
            )
        )


    positive_days = (
        return_series > 0
    ).sum()


    negative_days = (
        return_series < 0
    ).sum()


    win_rate = (
        positive_days
        / number_of_days
    )


    result = {
        "method": method_name,

        "start_date": (
            value_series.index[0]
        ),

        "end_date": (
            value_series.index[-1]
        ),

        "number_of_days": (
            number_of_days
        ),

        "final_value": (
            final_value
        ),

        "total_return_percent": (
            total_return * 100
        ),

        "annual_return_percent": (
            annual_return * 100
        ),

        "annual_volatility_percent": (
            annual_volatility * 100
        ),

        "sharpe_ratio": (
            sharpe_ratio
        ),

        "sortino_ratio": (
            sortino_ratio
        ),

        "maximum_drawdown_percent": (
            maximum_drawdown * 100
        ),

        "calmar_ratio": (
            calmar_ratio
        ),

        "win_rate_percent": (
            win_rate * 100
        ),

        "positive_days": int(
            positive_days
        ),

        "negative_days": int(
            negative_days
        )
    }


    return (
        result,
        drawdown_details
    )


# =====================================
# 5. 计算信息比率
# =====================================

def calculate_information_ratio(
    portfolio_return,
    benchmark_return
):

    active_return = (
        portfolio_return
        - benchmark_return
    )


    tracking_error = (
        active_return.std()
        * np.sqrt(
            TRADING_DAYS_PER_YEAR
        )
    )


    annual_active_return = (
        active_return.mean()
        * TRADING_DAYS_PER_YEAR
    )


    if tracking_error == 0:

        information_ratio = 0

    else:

        information_ratio = (
            annual_active_return
            / tracking_error
        )


    return (
        information_ratio,
        tracking_error
    )


# =====================================
# 6. 保存绩效结果
# =====================================

def save_performance_results(
    performance_summary,
    drawdown_summary
):

    save_performance = (
        performance_summary.copy()
    )


    save_performance[
        "start_date"
    ] = (
        save_performance[
            "start_date"
        ]
        .dt.strftime("%Y-%m-%d")
    )


    save_performance[
        "end_date"
    ] = (
        save_performance[
            "end_date"
        ]
        .dt.strftime("%Y-%m-%d")
    )


    save_drawdown = (
        drawdown_summary.copy()
    )


    date_columns = [
        "peak_date",
        "trough_date",
        "recovery_date"
    ]


    for column in date_columns:

        save_drawdown[column] = (
            pd.to_datetime(
                save_drawdown[column]
            )
            .dt.strftime("%Y-%m-%d")
        )


    connection = sqlite3.connect(
        DATABASE_PATH
    )


    save_performance.to_sql(
        "performance_summary",
        connection,
        if_exists="replace",
        index=False
    )


    save_drawdown.to_sql(
        "drawdown_summary",
        connection,
        if_exists="replace",
        index=False
    )


    connection.close()


    save_performance.to_csv(
        RESULT_DIR
        / "performance_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )


    save_drawdown.to_csv(
        RESULT_DIR
        / "drawdown_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )


# =====================================
# 7. 主程序
# =====================================

def main():

    print(
        "\nCalculating performance metrics..."
    )


    results = (
        load_backtest_results()
    )


    (
        portfolio_metrics,
        portfolio_drawdown
    ) = calculate_performance_metrics(
        "Multi-Factor Portfolio",
        results["portfolio_value"],
        results["daily_return"]
    )


    (
        benchmark_metrics,
        benchmark_drawdown
    ) = calculate_performance_metrics(
        "CSI 300 Benchmark",
        results["benchmark_value"],
        results["benchmark_return"]
    )


    performance_summary = pd.DataFrame(
        [
            portfolio_metrics,
            benchmark_metrics
        ]
    )


    drawdown_summary = pd.DataFrame(
        [
            {
                "method": (
                    "Multi-Factor Portfolio"
                ),
                **portfolio_drawdown
            },

            {
                "method": (
                    "CSI 300 Benchmark"
                ),
                **benchmark_drawdown
            }
        ]
    )


    (
        information_ratio,
        tracking_error
    ) = calculate_information_ratio(
        results["daily_return"],
        results["benchmark_return"]
    )


    save_performance_results(
        performance_summary,
        drawdown_summary
    )


    pd.set_option(
        "display.max_columns",
        None
    )

    pd.set_option(
        "display.width",
        240
    )

    pd.set_option(
        "display.float_format",
        "{:,.2f}".format
    )


    print("\n" + "=" * 110)

    print("PERFORMANCE SUMMARY")

    print("=" * 110)


    display_columns = [
        "method",
        "final_value",
        "total_return_percent",
        "annual_return_percent",
        "annual_volatility_percent",
        "sharpe_ratio",
        "sortino_ratio",
        "maximum_drawdown_percent",
        "calmar_ratio",
        "win_rate_percent"
    ]


    print(
        performance_summary[
            display_columns
        ].to_string(
            index=False
        )
    )


    print("\n" + "=" * 110)

    print("DRAWDOWN DETAILS")

    print("=" * 110)


    print(
        drawdown_summary.to_string(
            index=False
        )
    )


    print("\n" + "=" * 110)

    print("RELATIVE PERFORMANCE")

    print("=" * 110)


    print(
        f"Information ratio: "
        f"{information_ratio:.2f}"
    )


    print(
        f"Annual tracking error: "
        f"{tracking_error:.2%}"
    )


    print(
        f"Risk-free rate assumption: "
        f"{ANNUAL_RISK_FREE_RATE:.2%}"
    )


    print(
        "\nPerformance analysis completed."
    )


# =====================================
# 8. 启动程序
# =====================================

if __name__ == "__main__":

    main()