import sqlite3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config.settings import (
    DATABASE_PATH,
    CHART_DIR,
    RESULT_DIR,
    INITIAL_CAPITAL,
    TRADING_DAYS_PER_YEAR
)


# =====================================
# 1. 读取回测数据
# =====================================

def load_backtest_data():

    connection = sqlite3.connect(
        DATABASE_PATH
    )


    query = """
    SELECT
        date,
        portfolio_value,
        benchmark_value,
        daily_return,
        benchmark_return
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


    return results


# =====================================
# 2. 计算策略和基准回撤
# =====================================

def calculate_drawdowns(
    results
):

    portfolio_peak = (
        results[
            "portfolio_value"
        ].cummax()
    )


    portfolio_drawdown = (
        results[
            "portfolio_value"
        ]
        / portfolio_peak
        - 1
    )


    benchmark_peak = (
        results[
            "benchmark_value"
        ].cummax()
    )


    benchmark_drawdown = (
        results[
            "benchmark_value"
        ]
        / benchmark_peak
        - 1
    )


    return (
        portfolio_drawdown,
        benchmark_drawdown
    )


# =====================================
# 3. 计算每年收益率
# =====================================

def calculate_annual_returns(
    results
):

    daily_returns = results[
        [
            "daily_return",
            "benchmark_return"
        ]
    ].copy()


    annual_returns = (
        (
            1
            + daily_returns
        )
        .groupby(
            daily_returns.index.year
        )
        .prod()
        - 1
    )


    annual_returns.columns = [
        "Multi-Factor Portfolio",
        "CSI 300 Benchmark"
    ]


    annual_returns.index.name = "year"


    return annual_returns


# =====================================
# 4. 计算60日滚动波动率
# =====================================

def calculate_rolling_volatility(
    results
):

    rolling_volatility = (
        results[
            [
                "daily_return",
                "benchmark_return"
            ]
        ]
        .rolling(
            window=60
        )
        .std()
        * np.sqrt(
            TRADING_DAYS_PER_YEAR
        )
    )


    rolling_volatility.columns = [
        "Multi-Factor Portfolio",
        "CSI 300 Benchmark"
    ]


    return rolling_volatility


# =====================================
# 5. 保存年度收益率
# =====================================

def save_annual_returns(
    annual_returns
):

    save_data = (
        annual_returns
        .reset_index()
    )


    save_data.to_csv(
        RESULT_DIR
        / "annual_returns.csv",
        index=False,
        encoding="utf-8-sig"
    )


    connection = sqlite3.connect(
        DATABASE_PATH
    )


    save_data.to_sql(
        "annual_returns",
        connection,
        if_exists="replace",
        index=False
    )


    connection.close()


# =====================================
# 6. 创建完整回测图表
# =====================================

def create_backtest_charts(
    results,
    portfolio_drawdown,
    benchmark_drawdown,
    annual_returns,
    rolling_volatility
):

    plt.style.use(
        "ggplot"
    )


    figure, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(16, 10)
    )


    # ---------------------------------
    # 图1：累计净值
    # ---------------------------------

    normalized_portfolio = (
        results[
            "portfolio_value"
        ]
        / INITIAL_CAPITAL
        * 100
    )


    normalized_benchmark = (
        results[
            "benchmark_value"
        ]
        / INITIAL_CAPITAL
        * 100
    )


    axes[0, 0].plot(
        normalized_portfolio.index,
        normalized_portfolio,
        label="Multi-Factor Portfolio",
        linewidth=2
    )


    axes[0, 0].plot(
        normalized_benchmark.index,
        normalized_benchmark,
        label="CSI 300 Benchmark",
        linewidth=2
    )


    axes[0, 0].axhline(
        y=100,
        color="black",
        linestyle="--",
        linewidth=1,
        alpha=0.6
    )


    axes[0, 0].set_title(
        "Cumulative Portfolio Value"
    )


    axes[0, 0].set_xlabel(
        "Date"
    )


    axes[0, 0].set_ylabel(
        "Initial value = 100"
    )


    axes[0, 0].legend()


    # ---------------------------------
    # 图2：回撤曲线
    # ---------------------------------

    axes[0, 1].plot(
        portfolio_drawdown.index,
        portfolio_drawdown * 100,
        label="Multi-Factor Portfolio",
        linewidth=1.8
    )


    axes[0, 1].plot(
        benchmark_drawdown.index,
        benchmark_drawdown * 100,
        label="CSI 300 Benchmark",
        linewidth=1.8
    )


    axes[0, 1].fill_between(
        portfolio_drawdown.index,
        portfolio_drawdown * 100,
        0,
        alpha=0.15
    )


    axes[0, 1].axhline(
        y=0,
        color="black",
        linewidth=1
    )


    axes[0, 1].set_title(
        "Drawdown"
    )


    axes[0, 1].set_xlabel(
        "Date"
    )


    axes[0, 1].set_ylabel(
        "Drawdown (%)"
    )


    axes[0, 1].legend()


    # ---------------------------------
    # 图3：年度收益率
    # ---------------------------------

    (
        annual_returns
        * 100
    ).plot(
        kind="bar",
        ax=axes[1, 0],
        width=0.75
    )


    axes[1, 0].axhline(
        y=0,
        color="black",
        linewidth=1
    )


    axes[1, 0].set_title(
        "Annual Return"
    )


    axes[1, 0].set_xlabel(
        "Year"
    )


    axes[1, 0].set_ylabel(
        "Return (%)"
    )


    axes[1, 0].tick_params(
        axis="x",
        rotation=0
    )


    axes[1, 0].legend()


    # ---------------------------------
    # 图4：60日滚动波动率
    # ---------------------------------

    axes[1, 1].plot(
        rolling_volatility.index,
        rolling_volatility[
            "Multi-Factor Portfolio"
        ] * 100,
        label="Multi-Factor Portfolio",
        linewidth=1.5
    )


    axes[1, 1].plot(
        rolling_volatility.index,
        rolling_volatility[
            "CSI 300 Benchmark"
        ] * 100,
        label="CSI 300 Benchmark",
        linewidth=1.5
    )


    axes[1, 1].set_title(
        "60-Day Rolling Annualized Volatility"
    )


    axes[1, 1].set_xlabel(
        "Date"
    )


    axes[1, 1].set_ylabel(
        "Annualized volatility (%)"
    )


    axes[1, 1].legend()


    # ---------------------------------
    # 整张报告标题和排版
    # ---------------------------------

    figure.suptitle(
        "A-Share Multi-Factor Strategy Backtest",
        fontsize=18,
        fontweight="bold"
    )


    plt.tight_layout(
        rect=[
            0,
            0,
            1,
            0.96
        ]
    )


    output_path = (
        CHART_DIR
        / "backtest_report.png"
    )


    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )


    print(
        "Chart saved to:",
        output_path
    )


    plt.show()


# =====================================
# 7. 主程序
# =====================================

def main():

    print(
        "\nCreating backtest charts..."
    )


    results = (
        load_backtest_data()
    )


    (
        portfolio_drawdown,
        benchmark_drawdown
    ) = calculate_drawdowns(
        results
    )


    annual_returns = (
        calculate_annual_returns(
            results
        )
    )


    rolling_volatility = (
        calculate_rolling_volatility(
            results
        )
    )


    save_annual_returns(
        annual_returns
    )


    create_backtest_charts(
        results,
        portfolio_drawdown,
        benchmark_drawdown,
        annual_returns,
        rolling_volatility
    )


    print(
        "\nVisualization completed."
    )


# =====================================
# 8. 启动程序
# =====================================

if __name__ == "__main__":

    main()