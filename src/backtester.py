import sqlite3

import pandas as pd

from config.settings import (
    DATABASE_PATH,
    RESULT_DIR,
    STOCK_POOL,
    BENCHMARK_SYMBOL,
    INITIAL_CAPITAL,
    BUY_COMMISSION_RATE,
    SELL_COMMISSION_RATE,
    STAMP_DUTY_RATE,
    SLIPPAGE_RATE
)


# =====================================
# 1. 读取股票价格
# =====================================

def load_price_data():

    connection = sqlite3.connect(
        DATABASE_PATH
    )


    query = """
    SELECT
        symbol,
        date,
        close
    FROM daily_prices
    ORDER BY date, symbol;
    """


    price_data = pd.read_sql_query(
        query,
        connection
    )


    connection.close()


    if price_data.empty:

        raise ValueError(
            "No price data found."
        )


    price_data["date"] = (
        pd.to_datetime(
            price_data["date"]
        )
    )


    price_data["close"] = (
        pd.to_numeric(
            price_data["close"],
            errors="coerce"
        )
    )


    price_data = price_data.dropna(
        subset=[
            "symbol",
            "date",
            "close"
        ]
    )


    return price_data


# =====================================
# 2. 读取调仓指令
# =====================================

def load_rebalance_orders():

    connection = sqlite3.connect(
        DATABASE_PATH
    )


    query = """
    SELECT
        signal_date,
        execution_date,
        symbol,
        factor_rank,
        composite_score,
        target_weight
    FROM rebalance_orders
    ORDER BY signal_date, factor_rank;
    """


    orders = pd.read_sql_query(
        query,
        connection
    )


    connection.close()


    if orders.empty:

        raise ValueError(
            "No rebalance orders found. "
            "Please run "
            "python -m src.strategy first."
        )


    orders["signal_date"] = (
        pd.to_datetime(
            orders["signal_date"]
        )
    )


    orders["execution_date"] = (
        pd.to_datetime(
            orders["execution_date"]
        )
    )


    return orders


# =====================================
# 3. 建立股票收盘价矩阵
# =====================================

def create_stock_price_matrix(
    price_data
):

    stock_symbols = list(
        STOCK_POOL.keys()
    )


    stock_prices = price_data[
        price_data["symbol"].isin(
            stock_symbols
        )
    ].copy()


    price_matrix = (
        stock_prices
        .pivot_table(
            index="date",
            columns="symbol",
            values="close",
            aggfunc="last"
        )
        .sort_index()
    )


    if price_matrix.empty:

        raise ValueError(
            "Stock price matrix is empty."
        )


    return price_matrix


# =====================================
# 4. 读取基准指数价格
# =====================================

def create_benchmark_series(
    price_data,
    trading_dates
):

    benchmark_data = price_data[
        price_data["symbol"]
        == BENCHMARK_SYMBOL
    ].copy()


    if benchmark_data.empty:

        raise ValueError(
            "Benchmark data not found."
        )


    benchmark_data = (
        benchmark_data
        .sort_values("date")
        .drop_duplicates(
            subset=["date"],
            keep="last"
        )
    )


    benchmark_close = (
        benchmark_data
        .set_index("date")["close"]
        .reindex(trading_dates)
        .ffill()
    )


    return benchmark_close


# =====================================
# 5. 把调仓指令转换成每日权重
# =====================================

def create_daily_weights(
    orders,
    trading_dates,
    stock_symbols
):

    # ---------------------------------
    # 把月度选股指令转换成权重矩阵
    # ---------------------------------

    signal_weights = (
        orders
        .pivot_table(
            index="signal_date",
            columns="symbol",
            values="target_weight",
            aggfunc="sum"
        )
    )


    # ---------------------------------
    # 保证所有股票列都存在
    # ---------------------------------

    signal_weights = (
        signal_weights
        .reindex(
            columns=stock_symbols
        )
    )


    # 非入选股票必须明确设置为0
    # 不能让它保持NaN
    signal_weights = (
        signal_weights
        .fillna(0.0)
    )


    # ---------------------------------
    # 把月度信号放入每日交易日期
    # ---------------------------------

    target_weights = (
        signal_weights
        .reindex(
            trading_dates
        )
    )


    # 只有非调仓日才能延续上次权重
    target_weights = (
        target_weights
        .ffill()
        .fillna(0.0)
    )


    # ---------------------------------
    # 信号延迟一个交易日执行
    # ---------------------------------

    daily_weights = (
        target_weights
        .shift(1)
        .fillna(0.0)
    )


    # ---------------------------------
    # 检查每日总仓位
    # ---------------------------------

    daily_total_weight = (
        daily_weights
        .sum(axis=1)
    )


    maximum_total_weight = (
        daily_total_weight.max()
    )


    print(
        f"Maximum portfolio weight: "
        f"{maximum_total_weight:.2%}"
    )


    invalid_dates = daily_total_weight[
        daily_total_weight > 1.000001
    ]


    if not invalid_dates.empty:

        print(
            "\nDates with invalid weights:"
        )

        print(
            invalid_dates.head(10)
        )

        raise ValueError(
            "Daily portfolio weight is "
            "greater than 100%."
        )


    return daily_weights


# =====================================
# 6. 计算交易成本
# =====================================

def calculate_transaction_costs(
    daily_weights
):

    # 今日权重减去昨日权重
    weight_changes = (
        daily_weights
        .diff()
        .fillna(
            daily_weights
        )
    )


    # 权重增加代表买入
    buy_turnover = (
        weight_changes
        .clip(lower=0)
        .sum(axis=1)
    )


    # 权重减少代表卖出
    sell_turnover = (
        -weight_changes
        .clip(upper=0)
        .sum(axis=1)
    )


    buy_cost_rate = (
        BUY_COMMISSION_RATE
        + SLIPPAGE_RATE
    )


    sell_cost_rate = (
        SELL_COMMISSION_RATE
        + STAMP_DUTY_RATE
        + SLIPPAGE_RATE
    )


    transaction_cost = (
        buy_turnover
        * buy_cost_rate
        + sell_turnover
        * sell_cost_rate
    )


    return (
        buy_turnover,
        sell_turnover,
        transaction_cost
    )


# =====================================
# 7. 执行组合回测
# =====================================

def run_backtest(
    price_matrix,
    benchmark_close,
    daily_weights
):

    # ---------------------------------
    # 计算每只股票每日收益率
    # ---------------------------------

    stock_returns = (
        price_matrix
        .pct_change(
            fill_method=None
        )
        .fillna(0)
    )


    # ---------------------------------
    # 计算组合扣费前收益率
    # ---------------------------------

    gross_portfolio_return = (
        daily_weights
        * stock_returns
    ).sum(axis=1)


    # ---------------------------------
    # 计算换手率和交易成本
    # ---------------------------------

    (
        buy_turnover,
        sell_turnover,
        transaction_cost
    ) = calculate_transaction_costs(
        daily_weights
    )


    # ---------------------------------
    # 计算组合扣费后收益率
    # ---------------------------------

    net_portfolio_return = (
        gross_portfolio_return
        - transaction_cost
    )


    # ---------------------------------
    # 计算基准收益率
    # ---------------------------------

    benchmark_return = (
        benchmark_close
        .pct_change(
            fill_method=None
        )
        .fillna(0)
    )


    # ---------------------------------
    # 找到策略第一次真正持仓的日期
    # ---------------------------------

    total_weight = (
        daily_weights.sum(
            axis=1
        )
    )


    active_dates = total_weight[
        total_weight > 0
    ]


    if active_dates.empty:

        raise ValueError(
            "The strategy never entered "
            "the market."
        )


    backtest_start_date = (
        active_dates.index[0]
    )


    # 从第一次持仓开始比较
    gross_portfolio_return = (
        gross_portfolio_return.loc[
            backtest_start_date:
        ]
    )


    net_portfolio_return = (
        net_portfolio_return.loc[
            backtest_start_date:
        ]
    )


    benchmark_return = (
        benchmark_return.loc[
            backtest_start_date:
        ]
    )


    buy_turnover = (
        buy_turnover.loc[
            backtest_start_date:
        ]
    )


    sell_turnover = (
        sell_turnover.loc[
            backtest_start_date:
        ]
    )


    transaction_cost = (
        transaction_cost.loc[
            backtest_start_date:
        ]
    )


    total_weight = (
        total_weight.loc[
            backtest_start_date:
        ]
    )


    active_weights = (
        daily_weights.loc[
            backtest_start_date:
        ]
    )


    # ---------------------------------
    # 计算组合净值
    # ---------------------------------

    portfolio_value = (
        INITIAL_CAPITAL
        * (
            1
            + net_portfolio_return
        ).cumprod()
    )


    # ---------------------------------
    # 计算基准净值
    # ---------------------------------

    benchmark_value = (
        INITIAL_CAPITAL
        * (
            1
            + benchmark_return
        ).cumprod()
    )


    # ---------------------------------
    # 计算组合历史最高值
    # ---------------------------------

    portfolio_peak = (
        portfolio_value.cummax()
    )


    # ---------------------------------
    # 计算组合回撤
    # ---------------------------------

    drawdown = (
        portfolio_value
        / portfolio_peak
        - 1
    )


    # ---------------------------------
    # 估算现金金额
    # ---------------------------------

    cash_weight = (
        1
        - total_weight
    ).clip(
        lower=0
    )


    cash_value = (
        portfolio_value
        * cash_weight
    )


    # ---------------------------------
    # 整理回测结果
    # ---------------------------------

    results = pd.DataFrame(
        {
            "portfolio_value": (
                portfolio_value
            ),

            "benchmark_value": (
                benchmark_value
            ),

            "cash": (
                cash_value
            ),

            "gross_return": (
                gross_portfolio_return
            ),

            "daily_return": (
                net_portfolio_return
            ),

            "benchmark_return": (
                benchmark_return
            ),

            "buy_turnover": (
                buy_turnover
            ),

            "sell_turnover": (
                sell_turnover
            ),

            "transaction_cost": (
                transaction_cost
            ),

            "total_weight": (
                total_weight
            ),

            "drawdown": (
                drawdown
            )
        }
    )


    results.index.name = "date"


    return (
        results,
        active_weights
    )


# =====================================
# 8. 保存回测结果
# =====================================

def save_backtest_results(
    results,
    daily_weights
):

    save_results = (
        results
        .reset_index()
        .copy()
    )


    save_results["date"] = (
        save_results["date"]
        .dt.strftime("%Y-%m-%d")
    )


    connection = sqlite3.connect(
        DATABASE_PATH
    )


    save_results.to_sql(
        "backtest_results",
        connection,
        if_exists="replace",
        index=False
    )


    # ---------------------------------
    # 把每日持仓权重转换成长表
    # ---------------------------------

    weights_to_save = (
        daily_weights
        .stack()
        .reset_index()
    )


    weights_to_save.columns = [
        "date",
        "symbol",
        "weight"
    ]


    # 只保存真正持仓的记录
    weights_to_save = weights_to_save[
        weights_to_save["weight"] > 0
    ].copy()


    weights_to_save["date"] = (
        weights_to_save["date"]
        .dt.strftime("%Y-%m-%d")
    )


    weights_to_save.to_sql(
        "daily_weights",
        connection,
        if_exists="replace",
        index=False
    )


    connection.close()


    # ---------------------------------
    # 另外保存CSV
    # ---------------------------------

    save_results.to_csv(
        RESULT_DIR
        / "backtest_results.csv",
        index=False,
        encoding="utf-8-sig"
    )


    weights_to_save.to_csv(
        RESULT_DIR
        / "daily_weights.csv",
        index=False,
        encoding="utf-8-sig"
    )


    print(
        "Backtest results saved."
    )


# =====================================
# 9. 显示简单回测结果
# =====================================

def show_basic_results(
    results
):

    first_date = (
        results.index[0]
    )


    last_date = (
        results.index[-1]
    )


    final_portfolio_value = (
        results[
            "portfolio_value"
        ].iloc[-1]
    )


    final_benchmark_value = (
        results[
            "benchmark_value"
        ].iloc[-1]
    )


    portfolio_return = (
        final_portfolio_value
        / INITIAL_CAPITAL
        - 1
    )


    benchmark_return = (
        final_benchmark_value
        / INITIAL_CAPITAL
        - 1
    )


    excess_return = (
        portfolio_return
        - benchmark_return
    )


    total_cost = (
        results[
            "transaction_cost"
        ].sum()
    )


    print("\n" + "=" * 80)

    print("BASIC BACKTEST RESULTS")

    print("=" * 80)


    print(
        "Backtest period:",
        first_date.date(),
        "to",
        last_date.date()
    )


    print(
        f"Initial capital: "
        f"¥{INITIAL_CAPITAL:,.2f}"
    )


    print(
        f"Final portfolio value: "
        f"¥{final_portfolio_value:,.2f}"
    )


    print(
        f"Final benchmark value: "
        f"¥{final_benchmark_value:,.2f}"
    )


    print(
        f"Portfolio return: "
        f"{portfolio_return:.2%}"
    )


    print(
        f"Benchmark return: "
        f"{benchmark_return:.2%}"
    )


    print(
        f"Excess return: "
        f"{excess_return:.2%}"
    )


    print(
        f"Maximum drawdown: "
        f"{results['drawdown'].min():.2%}"
    )


    print(
        f"Total simplified cost: "
        f"{total_cost:.2%}"
    )


# =====================================
# 10. 主程序
# =====================================

def main():

    print(
        "\nRunning portfolio backtest..."
    )


    price_data = (
        load_price_data()
    )


    orders = (
        load_rebalance_orders()
    )


    price_matrix = (
        create_stock_price_matrix(
            price_data
        )
    )


    benchmark_close = (
        create_benchmark_series(
            price_data,
            price_matrix.index
        )
    )


    daily_weights = (
        create_daily_weights(
            orders,
            price_matrix.index,
            price_matrix.columns
        )
    )


    (
        results,
        active_weights
    ) = run_backtest(
        price_matrix,
        benchmark_close,
        daily_weights
    )


    save_backtest_results(
        results,
        active_weights
    )


    show_basic_results(
        results
    )


    print(
        "\nPortfolio backtest completed."
    )


# =====================================
# 11. 启动程序
# =====================================

if __name__ == "__main__":

    main()