import sqlite3

import numpy as np
import pandas as pd

from config.settings import (
    DATABASE_PATH,
    BENCHMARK_SYMBOL,
    MOMENTUM_SHORT_DAYS,
    MOMENTUM_LONG_DAYS,
    REVERSAL_DAYS,
    VOLATILITY_DAYS,
    SHORT_MA_DAYS,
    LONG_MA_DAYS,
    VOLUME_DAYS,
    TRADING_DAYS_PER_YEAR
)


# =====================================
# 1. 从数据库读取股票行情
# =====================================

def load_stock_prices():

    connection = sqlite3.connect(
        DATABASE_PATH
    )


    query = """
    SELECT
        symbol,
        date,
        close,
        volume
    FROM daily_prices
    WHERE symbol != ?
    ORDER BY symbol, date;
    """


    price_data = pd.read_sql_query(
        query,
        connection,
        params=(
            BENCHMARK_SYMBOL,
        )
    )


    connection.close()


    if price_data.empty:

        raise ValueError(
            "No stock price data found. "
            "Please run download_data.py first."
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


    price_data["volume"] = (
        pd.to_numeric(
            price_data["volume"],
            errors="coerce"
        )
    )


    price_data = price_data.dropna(
        subset=[
            "symbol",
            "date",
            "close",
            "volume"
        ]
    )


    return price_data


# =====================================
# 2. 计算单只股票的所有因子
# =====================================

def calculate_single_stock_factors(
    stock_data
):

    stock_data = stock_data.copy()


    stock_data = stock_data.sort_values(
        "date"
    )


    stock_data = stock_data.reset_index(
        drop=True
    )


    # ---------------------------------
    # 股票每日收益率
    # ---------------------------------

    stock_data["daily_return"] = (
        stock_data["close"]
        .pct_change()
    )


    # ---------------------------------
    # 因子1：20日动量
    # ---------------------------------

    stock_data["momentum_20"] = (
        stock_data["close"]
        / stock_data["close"].shift(
            MOMENTUM_SHORT_DAYS
        )
        - 1
    )


    # ---------------------------------
    # 因子2：60日动量
    # ---------------------------------

    stock_data["momentum_60"] = (
        stock_data["close"]
        / stock_data["close"].shift(
            MOMENTUM_LONG_DAYS
        )
        - 1
    )


    # ---------------------------------
    # 因子3：5日反转
    # ---------------------------------

    five_day_return = (
        stock_data["close"]
        / stock_data["close"].shift(
            REVERSAL_DAYS
        )
        - 1
    )


    stock_data["reversal_5"] = (
        -five_day_return
    )


    # ---------------------------------
    # 因子4：20日年化波动率
    # ---------------------------------

    stock_data["volatility_20"] = (
        stock_data["daily_return"]
        .rolling(
            window=VOLATILITY_DAYS
        )
        .std()
        * np.sqrt(
            TRADING_DAYS_PER_YEAR
        )
    )


    # ---------------------------------
    # 因子5：均线趋势
    # ---------------------------------

    short_ma = (
        stock_data["close"]
        .rolling(
            window=SHORT_MA_DAYS
        )
        .mean()
    )


    long_ma = (
        stock_data["close"]
        .rolling(
            window=LONG_MA_DAYS
        )
        .mean()
    )


    stock_data["trend_score"] = (
        short_ma
        / long_ma
        - 1
    )


    # ---------------------------------
    # 因子6：成交量变化
    # ---------------------------------

    average_volume = (
        stock_data["volume"]
        .rolling(
            window=VOLUME_DAYS
        )
        .mean()
    )


    stock_data["volume_ratio"] = (
        stock_data["volume"]
        / average_volume
        - 1
    )


    return stock_data


# =====================================
# 3. 计算股票池的所有因子
# =====================================

def calculate_all_factors(
    price_data
):

    all_factor_data = []


    symbols = (
        price_data["symbol"]
        .unique()
    )


    for symbol in symbols:

        print(
            f"Calculating factors "
            f"for {symbol}..."
        )


        single_stock = price_data[
            price_data["symbol"]
            == symbol
        ].copy()


        calculated_data = (
            calculate_single_stock_factors(
                single_stock
            )
        )


        all_factor_data.append(
            calculated_data
        )


    factor_data = pd.concat(
        all_factor_data,
        ignore_index=True
    )


    factor_columns = [
        "momentum_20",
        "momentum_60",
        "reversal_5",
        "volatility_20",
        "trend_score",
        "volume_ratio"
    ]


    # 把无穷大转换为空值
    factor_data = factor_data.replace(
        [
            np.inf,
            -np.inf
        ],
        np.nan
    )


    # 删除因子不完整的日期
    factor_data = factor_data.dropna(
        subset=factor_columns
    )


    factor_data = factor_data[
        [
            "symbol",
            "date",
            "momentum_20",
            "momentum_60",
            "reversal_5",
            "volatility_20",
            "trend_score",
            "volume_ratio"
        ]
    ].copy()


    factor_data = factor_data.sort_values(
        [
            "date",
            "symbol"
        ]
    )


    factor_data = factor_data.reset_index(
        drop=True
    )


    return factor_data


# =====================================
# 4. 保存因子到数据库
# =====================================

def save_factor_data(
    factor_data
):

    save_data = factor_data.copy()


    save_data["date"] = (
        pd.to_datetime(
            save_data["date"]
        )
        .dt.strftime("%Y-%m-%d")
    )


    # 综合得分下一课再计算
    save_data["composite_score"] = None


    columns = [
        "symbol",
        "date",
        "momentum_20",
        "momentum_60",
        "reversal_5",
        "volatility_20",
        "trend_score",
        "volume_ratio",
        "composite_score"
    ]


    records = list(
        save_data[
            columns
        ].itertuples(
            index=False,
            name=None
        )
    )


    connection = sqlite3.connect(
        DATABASE_PATH
    )

    cursor = connection.cursor()


    # 删除旧因子，避免留下过期数据
    cursor.execute(
        """
        DELETE FROM factor_values;
        """
    )


    sql = """
    INSERT OR REPLACE INTO factor_values (
        symbol,
        date,
        momentum_20,
        momentum_60,
        reversal_5,
        volatility_20,
        trend_score,
        volume_ratio,
        composite_score
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
    """


    cursor.executemany(
        sql,
        records
    )


    connection.commit()

    connection.close()


    print(
        f"Saved {len(records)} factor rows."
    )


# =====================================
# 5. 显示因子数据库概况
# =====================================

def show_factor_summary():

    connection = sqlite3.connect(
        DATABASE_PATH
    )


    query = """
    SELECT
        symbol,
        MIN(date) AS start_date,
        MAX(date) AS end_date,
        COUNT(*) AS factor_rows
    FROM factor_values
    GROUP BY symbol
    ORDER BY symbol;
    """


    summary = pd.read_sql_query(
        query,
        connection
    )


    connection.close()


    print("\n" + "=" * 80)

    print("FACTOR DATABASE SUMMARY")

    print("=" * 80)

    print(
        summary.to_string(
            index=False
        )
    )


# =====================================
# 6. 主程序
# =====================================

def main():

    print("\n" + "=" * 80)

    print("FACTOR LIBRARY")

    print("=" * 80)


    price_data = (
        load_stock_prices()
    )


    print(
        "Price rows loaded:",
        len(price_data)
    )


    print(
        "Number of stocks:",
        price_data["symbol"].nunique()
    )


    factor_data = (
        calculate_all_factors(
            price_data
        )
    )


    save_factor_data(
        factor_data
    )


    show_factor_summary()


    print("\nFirst 10 factor rows:")


    pd.set_option(
        "display.max_columns",
        None
    )

    pd.set_option(
        "display.width",
        200
    )

    pd.set_option(
        "display.float_format",
        "{:,.4f}".format
    )


    print(
        factor_data.head(10).to_string(
            index=False
        )
    )


# =====================================
# 7. 启动程序
# =====================================

if __name__ == "__main__":

    main()