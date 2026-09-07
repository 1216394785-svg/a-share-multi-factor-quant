import sqlite3

import numpy as np
import pandas as pd

from config.settings import (
    DATABASE_PATH,
    TOP_N_STOCKS
)


# =====================================
# 1. 数据库查询辅助函数
# =====================================

def read_database(
    query
):

    connection = sqlite3.connect(
        DATABASE_PATH
    )


    data = pd.read_sql_query(
        query,
        connection
    )


    connection.close()


    return data


# =====================================
# 测试1：行情不得重复
# =====================================

def test_no_duplicate_prices():

    duplicate_prices = read_database(
        """
        SELECT
            symbol,
            date,
            COUNT(*) AS row_count
        FROM daily_prices
        GROUP BY symbol, date
        HAVING COUNT(*) > 1;
        """
    )


    assert duplicate_prices.empty


# =====================================
# 测试2：调仓日期顺序必须正确
# =====================================

def test_signal_before_execution():

    orders = read_database(
        """
        SELECT
            signal_date,
            execution_date
        FROM rebalance_orders;
        """
    )


    assert not orders.empty


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


    assert (
        orders["signal_date"]
        < orders["execution_date"]
    ).all()


# =====================================
# 测试3：同一期不得重复买同一股票
# =====================================

def test_no_duplicate_rebalance_orders():

    duplicates = read_database(
        """
        SELECT
            execution_date,
            symbol,
            COUNT(*) AS row_count
        FROM rebalance_orders
        GROUP BY execution_date, symbol
        HAVING COUNT(*) > 1;
        """
    )


    assert duplicates.empty


# =====================================
# 测试4：每期股票数量不得超过设置
# =====================================

def test_number_of_selected_stocks():

    order_count = read_database(
        """
        SELECT
            execution_date,
            COUNT(*) AS number_of_stocks
        FROM rebalance_orders
        GROUP BY execution_date;
        """
    )


    assert not order_count.empty


    assert (
        order_count[
            "number_of_stocks"
        ] <= TOP_N_STOCKS
    ).all()


# =====================================
# 测试5：每期总权重不得超过100%
# =====================================

def test_total_portfolio_weight():

    weight_summary = read_database(
        """
        SELECT
            execution_date,
            SUM(target_weight) AS total_weight
        FROM rebalance_orders
        GROUP BY execution_date;
        """
    )


    assert not weight_summary.empty


    assert (
        weight_summary[
            "total_weight"
        ] > 0
    ).all()


    assert (
        weight_summary[
            "total_weight"
        ] <= 1.000001
    ).all()


# =====================================
# 测试6：训练后的因子权重等于100%
# =====================================

def test_trained_factor_weights():

    factor_weights = read_database(
        """
        SELECT
            direction,
            trained_weight
        FROM trained_factor_weights;
        """
    )


    assert not factor_weights.empty


    total_weight = (
        factor_weights[
            "trained_weight"
        ].sum()
    )


    assert np.isclose(
        total_weight,
        1.0,
        atol=0.000001
    )


    assert (
        factor_weights[
            "direction"
        ].isin(
            [
                -1,
                1
            ]
        )
    ).all()


# =====================================
# 测试7：训练集和测试集不能重叠
# =====================================

def test_training_testing_separation():

    period_dates = read_database(
        """
        SELECT
            period,
            MIN(date) AS start_date,
            MAX(date) AS end_date
        FROM trained_composite_ic
        GROUP BY period;
        """
    )


    training_row = period_dates[
        period_dates["period"]
        == "Training"
    ]


    testing_row = period_dates[
        period_dates["period"]
        == "Testing"
    ]


    assert not training_row.empty

    assert not testing_row.empty


    training_end = pd.to_datetime(
        training_row[
            "end_date"
        ].iloc[0]
    )


    testing_start = pd.to_datetime(
        testing_row[
            "start_date"
        ].iloc[0]
    )


    assert training_end < testing_start


# =====================================
# 测试8：样本外回测结果不能为空
# =====================================

def test_oos_results_not_empty():

    results = read_database(
        """
        SELECT *
        FROM oos_backtest_results;
        """
    )


    assert not results.empty


# =====================================
# 测试9：样本外资产必须始终为正
# =====================================

def test_oos_values_are_positive():

    results = read_database(
        """
        SELECT
            trained_value,
            original_value,
            benchmark_value
        FROM oos_backtest_results;
        """
    )


    value_columns = [
        "trained_value",
        "original_value",
        "benchmark_value"
    ]


    assert not results[
        value_columns
    ].isna().any().any()


    assert (
        results[
            value_columns
        ] > 0
    ).all().all()


# =====================================
# 测试10：每日收益不能出现空值
# =====================================

def test_oos_returns_have_no_missing_values():

    results = read_database(
        """
        SELECT
            trained_return,
            original_return,
            benchmark_return
        FROM oos_backtest_results;
        """
    )


    return_columns = [
        "trained_return",
        "original_return",
        "benchmark_return"
    ]


    assert not results[
        return_columns
    ].isna().any().any()


# =====================================
# 测试11：单日收益必须处于合理范围
# =====================================

def test_daily_returns_are_reasonable():

    results = read_database(
        """
        SELECT
            trained_return,
            original_return,
            benchmark_return
        FROM oos_backtest_results;
        """
    )


    return_columns = [
        "trained_return",
        "original_return",
        "benchmark_return"
    ]


    assert (
        results[
            return_columns
        ].abs() < 0.30
    ).all().all()