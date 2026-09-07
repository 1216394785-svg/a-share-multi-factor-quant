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
# 1. 读取样本外回测数据
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
# 2. 计算三种方法的回撤
# =====================================

def calculate_drawdowns(
    results
):

    value_columns = {
        "Trained Factor": (
            "trained_value"
        ),

        "Original Factor": (
            "original_value"
        ),

        "CSI 300": (
            "benchmark_value"
        )
    }


    drawdowns = pd.DataFrame(
        index=results.index
    )


    for method_name, value_column in (
        value_columns.items()
    ):

        value_series = results[
            value_column
        ]


        running_peak = (
            value_series.cummax()
        )


        drawdowns[
            method_name
        ] = (
            value_series
            / running_peak
            - 1
        )


    return drawdowns


# =====================================
# 3. 计算年度收益
# =====================================

def calculate_annual_returns(
    results
):

    return_data = results[
        [
            "trained_return",
            "original_return",
            "benchmark_return"
        ]
    ].copy()


    annual_returns = (
        (
            1
            + return_data
        )
        .groupby(
            return_data.index.year
        )
        .prod()
        - 1
    )


    annual_returns.columns = [
        "Trained Factor",
        "Original Factor",
        "CSI 300"
    ]


    annual_returns.index.name = "Year"


    return annual_returns


# =====================================
# 4. 计算60日滚动波动率
# =====================================

def calculate_rolling_volatility(
    results
):

    return_data = results[
        [
            "trained_return",
            "original_return",
            "benchmark_return"
        ]
    ].copy()


    rolling_volatility = (
        return_data
        .rolling(
            window=60
        )
        .std()
        * np.sqrt(
            TRADING_DAYS_PER_YEAR
        )
    )


    rolling_volatility.columns = [
        "Trained Factor",
        "Original Factor",
        "CSI 300"
    ]


    return rolling_volatility


# =====================================
# 5. 保存年度收益
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
        / "oos_annual_returns.csv",
        index=False,
        encoding="utf-8-sig"
    )


    connection = sqlite3.connect(
        DATABASE_PATH
    )


    save_data.to_sql(
        "oos_annual_returns",
        connection,
        if_exists="replace",
        index=False
    )


    connection.close()


# =====================================
# 6. 创建样本外报告图
# =====================================

def create_oos_charts(
    results,
    drawdowns,
    annual_returns,
    rolling_volatility
):

    plt.style.use(
        "seaborn-v0_8-whitegrid"
    )


    trained_color = "#1F4E79"

    original_color = "#D9822B"

    benchmark_color = "#4C956C"


    figure, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(16, 10)
    )


    # ---------------------------------
    # 图1：累计净值
    # ---------------------------------

    axes[0, 0].plot(
        results.index,
        (
            results["trained_value"]
            / INITIAL_CAPITAL
            * 100
        ),
        label="Trained Factor",
        color=trained_color,
        linewidth=2.2
    )


    axes[0, 0].plot(
        results.index,
        (
            results["original_value"]
            / INITIAL_CAPITAL
            * 100
        ),
        label="Original Factor",
        color=original_color,
        linewidth=1.8
    )


    axes[0, 0].plot(
        results.index,
        (
            results["benchmark_value"]
            / INITIAL_CAPITAL
            * 100
        ),
        label="CSI 300",
        color=benchmark_color,
        linewidth=2.0
    )


    axes[0, 0].axhline(
        y=100,
        color="black",
        linestyle="--",
        linewidth=1,
        alpha=0.5
    )


    axes[0, 0].set_title(
        "Out-of-Sample Portfolio Value"
    )


    axes[0, 0].set_xlabel(
        "Date"
    )


    axes[0, 0].set_ylabel(
        "Initial value = 100"
    )


    axes[0, 0].legend()


    # ---------------------------------
    # 图2：回撤
    # ---------------------------------

    axes[0, 1].plot(
        drawdowns.index,
        (
            drawdowns[
                "Trained Factor"
            ] * 100
        ),
        label="Trained Factor",
        color=trained_color,
        linewidth=2
    )


    axes[0, 1].plot(
        drawdowns.index,
        (
            drawdowns[
                "Original Factor"
            ] * 100
        ),
        label="Original Factor",
        color=original_color,
        linewidth=1.6
    )


    axes[0, 1].plot(
        drawdowns.index,
        (
            drawdowns[
                "CSI 300"
            ] * 100
        ),
        label="CSI 300",
        color=benchmark_color,
        linewidth=1.8
    )


    axes[0, 1].axhline(
        y=0,
        color="black",
        linewidth=1
    )


    axes[0, 1].fill_between(
        drawdowns.index,
        (
            drawdowns[
                "Trained Factor"
            ] * 100
        ),
        0,
        color=trained_color,
        alpha=0.12
    )


    axes[0, 1].set_title(
        "Out-of-Sample Drawdown"
    )


    axes[0, 1].set_xlabel(
        "Date"
    )


    axes[0, 1].set_ylabel(
        "Drawdown (%)"
    )


    axes[0, 1].legend()


    # ---------------------------------
    # 图3：年度收益
    # ---------------------------------

    (
        annual_returns
        * 100
    ).plot(
        kind="bar",
        ax=axes[1, 0],
        color=[
            trained_color,
            original_color,
            benchmark_color
        ],
        width=0.75
    )


    axes[1, 0].axhline(
        y=0,
        color="black",
        linewidth=1
    )


    axes[1, 0].set_title(
        "Out-of-Sample Annual Return"
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
        (
            rolling_volatility[
                "Trained Factor"
            ] * 100
        ),
        label="Trained Factor",
        color=trained_color,
        linewidth=2
    )


    axes[1, 1].plot(
        rolling_volatility.index,
        (
            rolling_volatility[
                "Original Factor"
            ] * 100
        ),
        label="Original Factor",
        color=original_color,
        linewidth=1.6
    )


    axes[1, 1].plot(
        rolling_volatility.index,
        (
            rolling_volatility[
                "CSI 300"
            ] * 100
        ),
        label="CSI 300",
        color=benchmark_color,
        linewidth=1.8
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
    # 整体标题和保存
    # ---------------------------------

    figure.suptitle(
        (
            "A-Share Multi-Factor Strategy "
            "— True Out-of-Sample Results"
        ),
        fontsize=17,
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
        / "oos_backtest_report.png"
    )


    plt.savefig(
        output_path,
        dpi=220,
        bbox_inches="tight"
    )


    print(
        "Out-of-sample chart saved to:",
        output_path
    )


    plt.show()


# =====================================
# 7. 主程序
# =====================================

def main():

    print(
        "\nCreating out-of-sample charts..."
    )


    results = (
        load_oos_results()
    )


    drawdowns = (
        calculate_drawdowns(
            results
        )
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


    create_oos_charts(
        results,
        drawdowns,
        annual_returns,
        rolling_volatility
    )


    print(
        "\nOut-of-sample visualization "
        "completed."
    )


# =====================================
# 8. 启动程序
# =====================================

if __name__ == "__main__":

    main()