import sqlite3

import numpy as np
import pandas as pd

from config.settings import (
    DATABASE_PATH,
    RESULT_DIR,
    INITIAL_CAPITAL,
    TOP_N_STOCKS,
    MAX_STOCK_WEIGHT,
    TRADING_DAYS_PER_YEAR
)

from src.backtester import (
    load_price_data,
    create_stock_price_matrix,
    create_benchmark_series,
    calculate_transaction_costs
)


# =====================================
# 1. 无风险利率假设
# =====================================

ANNUAL_RISK_FREE_RATE = 0.02


# =====================================
# 2. 读取训练后的因子权重
# =====================================

def load_trained_weights():

    connection = sqlite3.connect(
        DATABASE_PATH
    )


    query = """
    SELECT
        factor_column,
        factor_name,
        direction,
        trained_weight
    FROM trained_factor_weights;
    """


    trained_weights = pd.read_sql_query(
        query,
        connection
    )


    connection.close()


    if trained_weights.empty:

        raise ValueError(
            "No trained factor weights found. "
            "Please run "
            "python -m src.factor_validation first."
        )


    return trained_weights


# =====================================
# 3. 读取因子评分
# =====================================

def load_factor_scores():

    connection = sqlite3.connect(
        DATABASE_PATH
    )


    query = """
    SELECT
        symbol,
        date,
        momentum_20_score,
        momentum_60_score,
        reversal_5_score,
        volatility_20_score,
        trend_score_normalized,
        volume_ratio_score,
        composite_score,
        factor_rank
    FROM factor_scores
    ORDER BY date, symbol;
    """


    factor_scores = pd.read_sql_query(
        query,
        connection
    )


    connection.close()


    factor_scores["date"] = (
        pd.to_datetime(
            factor_scores["date"]
        )
    )


    return factor_scores


# =====================================
# 4. 读取测试期开始日期
# =====================================

def load_testing_start_date():

    connection = sqlite3.connect(
        DATABASE_PATH
    )


    query = """
    SELECT MIN(date) AS testing_start_date
    FROM trained_composite_ic
    WHERE period = 'Testing';
    """


    result = pd.read_sql_query(
        query,
        connection
    )


    connection.close()


    testing_start_value = (
        result.loc[
            0,
            "testing_start_date"
        ]
    )


    if pd.isna(
        testing_start_value
    ):

        raise ValueError(
            "Testing start date not found."
        )


    testing_start_date = (
        pd.to_datetime(
            testing_start_value
        )
    )


    return testing_start_date


# =====================================
# 5. 读取测试期月末信号日期
# =====================================

def load_testing_signal_dates(
    testing_start_date
):

    connection = sqlite3.connect(
        DATABASE_PATH
    )


    query = """
    SELECT DISTINCT signal_date
    FROM rebalance_orders
    WHERE signal_date >= ?
    ORDER BY signal_date;
    """


    signal_dates = pd.read_sql_query(
        query,
        connection,
        params=(
            testing_start_date.strftime(
                "%Y-%m-%d"
            ),
        )
    )


    connection.close()


    signal_dates[
        "signal_date"
    ] = (
        pd.to_datetime(
            signal_dates[
                "signal_date"
            ]
        )
    )


    return list(
        signal_dates[
            "signal_date"
        ]
    )


# =====================================
# 6. 计算训练后的综合得分
# =====================================

def calculate_trained_scores(
    factor_scores,
    trained_weights
):

    factor_scores = (
        factor_scores.copy()
    )


    factor_scores[
        "trained_composite_score"
    ] = 0.0


    for _, factor in (
        trained_weights.iterrows()
    ):

        factor_column = (
            factor[
                "factor_column"
            ]
        )


        direction = float(
            factor[
                "direction"
            ]
        )


        trained_weight = float(
            factor[
                "trained_weight"
            ]
        )


        factor_scores[
            "trained_composite_score"
        ] += (
            factor_scores[
                factor_column
            ]
            * direction
            * trained_weight
        )


    return factor_scores


# =====================================
# 7. 生成测试期选股指令
# =====================================

def generate_testing_orders(
    factor_scores,
    signal_dates
):

    all_orders = []


    for signal_date in signal_dates:

        daily_scores = factor_scores[
            factor_scores["date"]
            == signal_date
        ].copy()


        if daily_scores.empty:

            continue


        # ---------------------------------
        # 训练后的综合因子
        # ---------------------------------

        trained_selection = (
            daily_scores
            .sort_values(
                by="trained_composite_score",
                ascending=False
            )
            .head(TOP_N_STOCKS)
        )


        trained_weight = min(
            1
            / len(trained_selection),
            MAX_STOCK_WEIGHT
        )


        for _, stock in (
            trained_selection.iterrows()
        ):

            all_orders.append(
                {
                    "strategy": (
                        "Trained Factor"
                    ),

                    "signal_date": (
                        signal_date
                    ),

                    "symbol": (
                        stock["symbol"]
                    ),

                    "score": float(
                        stock[
                            "trained_composite_score"
                        ]
                    ),

                    "target_weight": (
                        trained_weight
                    )
                }
            )


        # ---------------------------------
        # 原始综合因子
        # ---------------------------------

        original_selection = (
            daily_scores
            .sort_values(
                by="composite_score",
                ascending=False
            )
            .head(TOP_N_STOCKS)
        )


        original_weight = min(
            1
            / len(original_selection),
            MAX_STOCK_WEIGHT
        )


        for _, stock in (
            original_selection.iterrows()
        ):

            all_orders.append(
                {
                    "strategy": (
                        "Original Factor"
                    ),

                    "signal_date": (
                        signal_date
                    ),

                    "symbol": (
                        stock["symbol"]
                    ),

                    "score": float(
                        stock[
                            "composite_score"
                        ]
                    ),

                    "target_weight": (
                        original_weight
                    )
                }
            )


    testing_orders = pd.DataFrame(
        all_orders
    )


    if testing_orders.empty:

        raise ValueError(
            "No testing orders generated."
        )


    return testing_orders


# =====================================
# 8. 转换成每日权重
# =====================================

def create_oos_daily_weights(
    testing_orders,
    strategy_name,
    trading_dates,
    stock_symbols
):

    strategy_orders = testing_orders[
        testing_orders["strategy"]
        == strategy_name
    ].copy()


    signal_weights = (
        strategy_orders
        .pivot_table(
            index="signal_date",
            columns="symbol",
            values="target_weight",
            aggfunc="sum"
        )
    )


    signal_weights = (
        signal_weights
        .reindex(
            columns=stock_symbols
        )
        .fillna(0.0)
    )


    target_weights = (
        signal_weights
        .reindex(
            trading_dates
        )
        .ffill()
        .fillna(0.0)
    )


    # 月末信号延迟一个交易日
    daily_weights = (
        target_weights
        .shift(1)
        .fillna(0.0)
    )


    total_weight = (
        daily_weights.sum(
            axis=1
        )
    )


    if (
        total_weight.max()
        > 1.000001
    ):

        raise ValueError(
            f"{strategy_name} weight "
            f"is greater than 100%."
        )


    return daily_weights


# =====================================
# 9. 计算单个策略每日收益
# =====================================

def calculate_strategy_returns(
    stock_returns,
    daily_weights
):

    gross_return = (
        daily_weights
        * stock_returns
    ).sum(axis=1)


    (
        buy_turnover,
        sell_turnover,
        transaction_cost
    ) = calculate_transaction_costs(
        daily_weights
    )


    net_return = (
        gross_return
        - transaction_cost
    )


    return {
        "gross_return": (
            gross_return
        ),

        "net_return": (
            net_return
        ),

        "transaction_cost": (
            transaction_cost
        ),

        "buy_turnover": (
            buy_turnover
        ),

        "sell_turnover": (
            sell_turnover
        )
    }


# =====================================
# 10. 计算绩效指标
# =====================================

def calculate_metrics(
    method_name,
    return_series,
    value_series,
    total_cost
):

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


    annual_volatility = (
        return_series.std()
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


    if return_series.std() == 0:

        sharpe_ratio = 0

    else:

        sharpe_ratio = (
            excess_daily_return.mean()
            / return_series.std()
            * np.sqrt(
                TRADING_DAYS_PER_YEAR
            )
        )


    drawdown = (
        value_series
        / value_series.cummax()
        - 1
    )


    maximum_drawdown = (
        drawdown.min()
    )


    return {
        "method": method_name,

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

        "maximum_drawdown_percent": (
            maximum_drawdown * 100
        ),

        "total_cost_percent": (
            total_cost * 100
        )
    }


# =====================================
# 11. 执行样本外回测
# =====================================

def run_oos_backtest(
    price_matrix,
    benchmark_close,
    trained_weights,
    original_weights
):

    stock_returns = (
        price_matrix
        .pct_change(
            fill_method=None
        )
        .fillna(0)
    )


    trained_results = (
        calculate_strategy_returns(
            stock_returns,
            trained_weights
        )
    )


    original_results = (
        calculate_strategy_returns(
            stock_returns,
            original_weights
        )
    )


    benchmark_return = (
        benchmark_close
        .pct_change(
            fill_method=None
        )
        .fillna(0)
    )


    trained_active = (
        trained_weights.sum(
            axis=1
        ) > 0
    )


    original_active = (
        original_weights.sum(
            axis=1
        ) > 0
    )


    common_active = (
        trained_active
        & original_active
    )


    if not common_active.any():

        raise ValueError(
            "No active testing dates found."
        )


    first_active_date = (
        common_active[
            common_active
        ].index[0]
    )


    trained_return = (
        trained_results[
            "net_return"
        ].loc[
            first_active_date:
        ]
    )


    original_return = (
        original_results[
            "net_return"
        ].loc[
            first_active_date:
        ]
    )


    benchmark_return = (
        benchmark_return.loc[
            first_active_date:
        ]
    )


    trained_cost = (
        trained_results[
            "transaction_cost"
        ].loc[
            first_active_date:
        ]
    )


    original_cost = (
        original_results[
            "transaction_cost"
        ].loc[
            first_active_date:
        ]
    )


    trained_value = (
        INITIAL_CAPITAL
        * (
            1 + trained_return
        ).cumprod()
    )


    original_value = (
        INITIAL_CAPITAL
        * (
            1 + original_return
        ).cumprod()
    )


    benchmark_value = (
        INITIAL_CAPITAL
        * (
            1 + benchmark_return
        ).cumprod()
    )


    results = pd.DataFrame(
        {
            "trained_return": (
                trained_return
            ),

            "original_return": (
                original_return
            ),

            "benchmark_return": (
                benchmark_return
            ),

            "trained_value": (
                trained_value
            ),

            "original_value": (
                original_value
            ),

            "benchmark_value": (
                benchmark_value
            ),

            "trained_transaction_cost": (
                trained_cost
            ),

            "original_transaction_cost": (
                original_cost
            )
        }
    )


    results.index.name = "date"


    return results


# =====================================
# 12. 保存样本外结果
# =====================================

def save_oos_results(
    results,
    summary,
    testing_orders
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


    save_orders = (
        testing_orders.copy()
    )


    save_orders["signal_date"] = (
        save_orders["signal_date"]
        .dt.strftime("%Y-%m-%d")
    )


    connection = sqlite3.connect(
        DATABASE_PATH
    )


    save_results.to_sql(
        "oos_backtest_results",
        connection,
        if_exists="replace",
        index=False
    )


    summary.to_sql(
        "oos_performance_summary",
        connection,
        if_exists="replace",
        index=False
    )


    save_orders.to_sql(
        "oos_testing_orders",
        connection,
        if_exists="replace",
        index=False
    )


    connection.close()


    save_results.to_csv(
        RESULT_DIR
        / "oos_backtest_results.csv",
        index=False,
        encoding="utf-8-sig"
    )


    summary.to_csv(
        RESULT_DIR
        / "oos_performance_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )


# =====================================
# 13. 主程序
# =====================================

def main():

    print(
        "\nRunning true out-of-sample backtest..."
    )


    trained_factor_weights = (
        load_trained_weights()
    )


    factor_scores = (
        load_factor_scores()
    )


    testing_start_date = (
        load_testing_start_date()
    )


    signal_dates = (
        load_testing_signal_dates(
            testing_start_date
        )
    )


    factor_scores = (
        calculate_trained_scores(
            factor_scores,
            trained_factor_weights
        )
    )


    testing_orders = (
        generate_testing_orders(
            factor_scores,
            signal_dates
        )
    )


    price_data = (
        load_price_data()
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


    trained_daily_weights = (
        create_oos_daily_weights(
            testing_orders,
            "Trained Factor",
            price_matrix.index,
            price_matrix.columns
        )
    )


    original_daily_weights = (
        create_oos_daily_weights(
            testing_orders,
            "Original Factor",
            price_matrix.index,
            price_matrix.columns
        )
    )


    results = (
        run_oos_backtest(
            price_matrix,
            benchmark_close,
            trained_daily_weights,
            original_daily_weights
        )
    )


    trained_metrics = (
        calculate_metrics(
            "Trained Factor",
            results["trained_return"],
            results["trained_value"],
            results[
                "trained_transaction_cost"
            ].sum()
        )
    )


    original_metrics = (
        calculate_metrics(
            "Original Factor",
            results["original_return"],
            results["original_value"],
            results[
                "original_transaction_cost"
            ].sum()
        )
    )


    benchmark_metrics = (
        calculate_metrics(
            "CSI 300 Benchmark",
            results["benchmark_return"],
            results["benchmark_value"],
            0
        )
    )


    summary = pd.DataFrame(
        [
            trained_metrics,
            original_metrics,
            benchmark_metrics
        ]
    )


    save_oos_results(
        results,
        summary,
        testing_orders
    )


    pd.set_option(
        "display.max_columns",
        None
    )

    pd.set_option(
        "display.width",
        220
    )

    pd.set_option(
        "display.float_format",
        "{:,.2f}".format
    )


    print("\n" + "=" * 110)

    print("TRUE OUT-OF-SAMPLE PERFORMANCE")

    print("=" * 110)


    print(
        "Testing start:",
        results.index[0].date()
    )


    print(
        "Testing end:",
        results.index[-1].date()
    )


    print("\n")


    print(
        summary.to_string(
            index=False
        )
    )


    print(
        "\nOut-of-sample backtest completed."
    )


# =====================================
# 14. 启动程序
# =====================================

if __name__ == "__main__":

    main()