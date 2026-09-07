import pandas as pd
import yfinance as yf

from config.settings import (
    START_DATE,
    END_DATE,
    STOCK_POOL
)

from src.database import (
    get_connection,
    initialize_database
)


# =====================================
# 1. 下载单只股票
# =====================================

def download_single_stock(
    symbol,
    stock_name
):

    print(
        f"Downloading {symbol} "
        f"({stock_name})..."
    )


    stock_data = yf.download(
        symbol,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=True,
        progress=False,
        threads=False
    )


    # 检查是否下载到数据
    if stock_data.empty:

        raise ValueError(
            f"No data downloaded for "
            f"{symbol}"
        )


    # 处理yfinance可能产生的双层列名
    if isinstance(
        stock_data.columns,
        pd.MultiIndex
    ):

        stock_data.columns = (
            stock_data.columns
            .get_level_values(0)
        )


    # 把日期从索引变成普通列
    stock_data = (
        stock_data
        .reset_index()
    )


    # 把全部列名转换成小写
    stock_data.columns = [
        str(column).lower()
        for column in stock_data.columns
    ]


    # 检查需要的列是否存在
    required_columns = [
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]


    missing_columns = [
        column
        for column in required_columns
        if column not in stock_data.columns
    ]


    if missing_columns:

        raise ValueError(
            f"Missing columns for {symbol}: "
            f"{missing_columns}"
        )


    # 只保留需要的数据
    stock_data = stock_data[
        required_columns
    ].copy()


    # 添加股票代码
    stock_data.insert(
        0,
        "symbol",
        symbol
    )


    # 整理日期格式
    stock_data["date"] = (
        pd.to_datetime(
            stock_data["date"]
        )
        .dt.strftime("%Y-%m-%d")
    )


    # 删除没有收盘价的数据
    stock_data = stock_data.dropna(
        subset=["close"]
    )


    # 按日期排列
    stock_data = stock_data.sort_values(
        "date"
    )


    # 重置行号
    stock_data = stock_data.reset_index(
        drop=True
    )


    print(
        f"Downloaded {len(stock_data)} "
        f"rows for {symbol}."
    )


    return stock_data


# =====================================
# 2. 下载整个股票池
# =====================================

def download_stock_pool():

    all_stock_data = []


    for symbol, stock_name in (
        STOCK_POOL.items()
    ):

        try:

            stock_data = (
                download_single_stock(
                    symbol,
                    stock_name
                )
            )

            all_stock_data.append(
                stock_data
            )


        except Exception as error:

            print(
                f"Failed to download "
                f"{symbol}: {error}"
            )


    if not all_stock_data:

        raise ValueError(
            "No stock data was downloaded."
        )


    combined_data = pd.concat(
        all_stock_data,
        ignore_index=True
    )


    return combined_data


# =====================================
# 3. 保存行情到SQLite
# =====================================

def save_daily_prices(
    price_data
):

    connection = get_connection()

    cursor = connection.cursor()


    sql = """
    INSERT OR REPLACE INTO daily_prices (
        symbol,
        date,
        open,
        high,
        low,
        close,
        volume
    )
    VALUES (?, ?, ?, ?, ?, ?, ?);
    """


    columns = [
        "symbol",
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]


    records = list(
        price_data[
            columns
        ].itertuples(
            index=False,
            name=None
        )
    )


    cursor.executemany(
        sql,
        records
    )


    connection.commit()

    connection.close()


    print(
        f"Saved {len(records)} rows "
        f"to the database."
    )


# =====================================
# 4. 查询某只股票保存了多少行
# =====================================

def count_saved_rows(
    symbol
):

    connection = get_connection()

    cursor = connection.cursor()


    cursor.execute(
        """
        SELECT COUNT(*)
        FROM daily_prices
        WHERE symbol = ?;
        """,
        (symbol,)
    )


    result = cursor.fetchone()

    connection.close()


    row_count = result[0]

    return row_count


# =====================================
# 5. 直接运行时先测试贵州茅台
# =====================================

if __name__ == "__main__":

    initialize_database()

    test_symbol = "600519.SS"

    test_name = (
        STOCK_POOL[test_symbol]
    )

    test_data = (
        download_single_stock(
            test_symbol,
            test_name
        )
    )

    save_daily_prices(
        test_data
    )

    saved_rows = count_saved_rows(
        test_symbol
    )

    print(
        "Test completed successfully."
    )

    print(
        "Symbol:",
        test_symbol
    )

    print(
        "Rows in database:",
        saved_rows
    )

    print(
        "\nFirst 5 rows:"
    )

    print(
        test_data.head().to_string(
            index=False
        )
    )