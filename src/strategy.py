import sqlite3

import pandas as pd

from config.settings import (
    DATABASE_PATH,
    RESULT_DIR,
    BENCHMARK_SYMBOL,
    TOP_N_STOCKS,
    MAX_STOCK_WEIGHT
)


# =====================================
# 1. 读取因子评分数据
# =====================================

def load_factor_scores():

    connection = sqlite3.connect(
        DATABASE_PATH
    )


    query = """
    SELECT
        symbol,
        date,
        composite_score,
        factor_rank
    FROM factor_scores
    ORDER BY date, factor_rank;
    """


    factor_data = pd.read_sql_query(
        query,
        connection
    )


    connection.close()


    if factor_data.empty:

        raise ValueError(
            "No factor scores found. "
            "Please run "
            "python -m src.factor_analysis first."
        )


    factor_data["date"] = (
        pd.to_datetime(
            factor_data["date"]
        )
    )


    return factor_data


# =====================================
# 2. 读取市场交易日期
# =====================================

def load_trading_dates():

    connection = sqlite3.connect(
        DATABASE_PATH
    )


    # 优先使用沪深300的交易日期
    query = """
    SELECT DISTINCT date
    FROM daily_prices
    WHERE symbol = ?
    ORDER BY date;
    """


    trading_dates = pd.read_sql_query(
        query,
        connection,
        params=(
            BENCHMARK_SYMBOL,
        )
    )


    # 如果没有指数数据
    # 就使用股票池的全部交易日期
    if trading_dates.empty:

        backup_query = """
        SELECT DISTINCT date
        FROM daily_prices
        ORDER BY date;
        """


        trading_dates = pd.read_sql_query(
            backup_query,
            connection
        )


    connection.close()


    if trading_dates.empty:

        raise ValueError(
            "No trading dates found."
        )


    trading_dates["date"] = (
        pd.to_datetime(
            trading_dates["date"]
        )
    )


    trading_dates = (
        trading_dates["date"]
        .sort_values()
        .drop_duplicates()
        .reset_index(drop=True)
    )


    return trading_dates


# =====================================
# 3. 找到每个月最后一个因子日期
# =====================================

def find_month_end_signal_dates(
    factor_data
):

    factor_data = factor_data.copy()


    factor_data["month"] = (
        factor_data["date"]
        .dt.to_period("M")
    )


    month_end_dates = (
        factor_data
        .groupby("month")["date"]
        .max()
        .sort_values()
    )


    # 最后一个月可能还没有结束
    # 因此从历史回测中删除最后一个月
    if len(month_end_dates) > 1:

        month_end_dates = (
            month_end_dates.iloc[:-1]
        )


    month_end_dates = (
        month_end_dates.tolist()
    )


    return month_end_dates


# =====================================
# 4. 找到信号后的下一个交易日
# =====================================

def find_next_trading_date(
    signal_date,
    trading_dates
):

    future_dates = trading_dates[
        trading_dates > signal_date
    ]


    if future_dates.empty:

        return None


    next_date = future_dates.iloc[0]

    return next_date


# =====================================
# 5. 生成月度调仓指令
# =====================================

def generate_rebalance_orders(
    factor_data,
    trading_dates
):

    signal_dates = (
        find_month_end_signal_dates(
            factor_data
        )
    )


    all_orders = []


    for signal_date in signal_dates:

        execution_date = (
            find_next_trading_date(
                signal_date,
                trading_dates
            )
        )


        if execution_date is None:

            continue


        daily_ranking = factor_data[
            factor_data["date"]
            == signal_date
        ].copy()


        daily_ranking = (
            daily_ranking
            .sort_values(
                by="factor_rank",
                ascending=True
            )
        )


        selected_stocks = (
            daily_ranking
            .head(TOP_N_STOCKS)
            .copy()
        )


        if selected_stocks.empty:

            continue


        equal_weight = (
            1
            / len(selected_stocks)
        )


        target_weight = min(
            equal_weight,
            MAX_STOCK_WEIGHT
        )


        for _, stock in (
            selected_stocks.iterrows()
        ):

            order = {
                "signal_date": (
                    signal_date
                ),

                "execution_date": (
                    execution_date
                ),

                "symbol": (
                    stock["symbol"]
                ),

                "factor_rank": int(
                    stock["factor_rank"]
                ),

                "composite_score": float(
                    stock[
                        "composite_score"
                    ]
                ),

                "target_weight": (
                    target_weight
                )
            }


            all_orders.append(
                order
            )


    rebalance_orders = pd.DataFrame(
        all_orders
    )


    if rebalance_orders.empty:

        raise ValueError(
            "No rebalance orders generated."
        )


    rebalance_orders = (
        rebalance_orders
        .sort_values(
            [
                "execution_date",
                "factor_rank"
            ]
        )
        .reset_index(drop=True)
    )


    return rebalance_orders


# =====================================
# 6. 检查每期权重
# =====================================

def validate_rebalance_orders(
    rebalance_orders
):

    # ---------------------------------
    # 删除同一天、同一股票的重复指令
    # ---------------------------------

    rebalance_orders.drop_duplicates(
        subset=[
            "execution_date",
            "symbol"
        ],
        keep="first",
        inplace=True
    )


    # ---------------------------------
    # 每个调仓日最多保留前N只股票
    # ---------------------------------

    rebalance_orders.sort_values(
        by=[
            "execution_date",
            "factor_rank"
        ],
        inplace=True
    )


    keep_rows = (
        rebalance_orders
        .groupby(
            "execution_date",
            group_keys=False
        )
        .head(TOP_N_STOCKS)
        .index
    )


    remove_rows = (
        rebalance_orders.index
        .difference(keep_rows)
    )


    rebalance_orders.drop(
        index=remove_rows,
        inplace=True
    )


    # ---------------------------------
    # 重新计算每个调仓日的股票数量
    # ---------------------------------

    rebalance_orders[
        "selected_count"
    ] = (
        rebalance_orders
        .groupby("execution_date")[
            "symbol"
        ]
        .transform("count")
    )


    # ---------------------------------
    # 重新计算等权目标权重
    # ---------------------------------

    rebalance_orders[
        "target_weight"
    ] = (
        1
        / rebalance_orders[
            "selected_count"
        ]
    )


    # 单只股票不得超过最大权重
    rebalance_orders[
        "target_weight"
    ] = (
        rebalance_orders[
            "target_weight"
        ]
        .clip(
            upper=MAX_STOCK_WEIGHT
        )
    )


    # 删除临时统计列
    rebalance_orders.drop(
        columns=[
            "selected_count"
        ],
        inplace=True
    )


    # ---------------------------------
    # 重新统计每期总权重
    # ---------------------------------

    weight_summary = (
        rebalance_orders
        .groupby("execution_date")
        .agg(
            number_of_stocks=(
                "symbol",
                "count"
            ),

            total_weight=(
                "target_weight",
                "sum"
            )
        )
        .reset_index()
    )


    # 处理小数计算误差
    weight_summary[
        "total_weight"
    ] = (
        weight_summary[
            "total_weight"
        ]
        .round(10)
    )


    # ---------------------------------
    # 最终安全检查
    # ---------------------------------

    invalid_weights = weight_summary[
        weight_summary["total_weight"]
        > 1.0
    ]


    if not invalid_weights.empty:

        print(
            "\nInvalid weight periods:"
        )

        print(
            invalid_weights.to_string(
                index=False
            )
        )

        raise ValueError(
            "Portfolio weight is greater "
            "than 100%."
        )


    return weight_summary
# =====================================
# 7. 保存调仓指令
# =====================================

def save_rebalance_orders(
    rebalance_orders
):

    save_data = rebalance_orders.copy()


    save_data["signal_date"] = (
        save_data["signal_date"]
        .dt.strftime("%Y-%m-%d")
    )


    save_data["execution_date"] = (
        save_data["execution_date"]
        .dt.strftime("%Y-%m-%d")
    )


    connection = sqlite3.connect(
        DATABASE_PATH
    )


    save_data.to_sql(
        "rebalance_orders",
        connection,
        if_exists="replace",
        index=False
    )


    connection.close()


    output_path = (
        RESULT_DIR
        / "rebalance_orders.csv"
    )


    save_data.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig"
    )


    print(
        "Saved rebalance orders to:",
        output_path
    )


# =====================================
# 8. 显示最近的调仓指令
# =====================================

def show_latest_orders(
    rebalance_orders
):

    latest_execution_date = (
        rebalance_orders[
            "execution_date"
        ].max()
    )


    latest_orders = rebalance_orders[
        rebalance_orders[
            "execution_date"
        ]
        == latest_execution_date
    ].copy()


    print("\n" + "=" * 90)

    print("LATEST REBALANCE ORDERS")

    print("=" * 90)

    print(
        "Execution date:",
        latest_execution_date.date()
    )


    print(
        latest_orders.to_string(
            index=False
        )
    )


# =====================================
# 9. 主程序
# =====================================

def main():

    print("\nGenerating rebalance orders...")


    factor_data = (
        load_factor_scores()
    )


    trading_dates = (
        load_trading_dates()
    )


    rebalance_orders = (
        generate_rebalance_orders(
            factor_data,
            trading_dates
        )
    )


    weight_summary = (
        validate_rebalance_orders(
            rebalance_orders
        )
    )


    save_rebalance_orders(
        rebalance_orders
    )


    print(
        "Number of rebalance periods:",
        rebalance_orders[
            "execution_date"
        ].nunique()
    )


    print(
        "Number of orders:",
        len(rebalance_orders)
    )


    print("\nWeight check:")


    print(
        weight_summary.tail().to_string(
            index=False
        )
    )


    show_latest_orders(
        rebalance_orders
    )


    print(
        "\nStrategy signal generation "
        "completed."
    )


# =====================================
# 10. 启动程序
# =====================================

if __name__ == "__main__":

    main()