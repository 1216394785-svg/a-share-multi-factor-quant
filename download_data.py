import pandas as pd

from config.settings import (
    BENCHMARK_SYMBOL,
    BENCHMARK_NAME
)

from src.database import (
    get_connection,
    initialize_database
)

from src.data_loader import (
    download_single_stock,
    download_stock_pool,
    save_daily_prices
)


# =====================================
# 1. 显示数据库中的行情概况
# =====================================

def show_database_summary():

    connection = get_connection()


    query = """
    SELECT
        symbol,
        MIN(date) AS start_date,
        MAX(date) AS end_date,
        COUNT(*) AS number_of_rows
    FROM daily_prices
    GROUP BY symbol
    ORDER BY symbol;
    """


    summary = pd.read_sql_query(
        query,
        connection
    )


    connection.close()


    print("\n" + "=" * 80)

    print("DATABASE SUMMARY")

    print("=" * 80)


    if summary.empty:

        print(
            "The database has no price data."
        )

    else:

        print(
            summary.to_string(
                index=False
            )
        )


    return summary


# =====================================
# 2. 主程序
# =====================================

def main():

    print("\n" + "=" * 80)

    print("A-SHARE MARKET DATA DOWNLOAD")

    print("=" * 80)


    # 创建数据库和数据表
    initialize_database()


    # ---------------------------------
    # 下载完整股票池
    # ---------------------------------

    print(
        "\nDownloading stock pool..."
    )


    stock_data = (
        download_stock_pool()
    )


    save_daily_prices(
        stock_data
    )


    # ---------------------------------
    # 下载沪深300基准
    # ---------------------------------

    print(
        "\nDownloading benchmark..."
    )


    try:

        benchmark_data = (
            download_single_stock(
                BENCHMARK_SYMBOL,
                BENCHMARK_NAME
            )
        )


        save_daily_prices(
            benchmark_data
        )


    except Exception as error:

        print(
            "Benchmark download failed:",
            error
        )


    # ---------------------------------
    # 显示数据库结果
    # ---------------------------------

    summary = (
        show_database_summary()
    )


    number_of_symbols = (
        len(summary)
    )


    total_rows = int(
        summary[
            "number_of_rows"
        ].sum()
    )


    print("\n" + "=" * 80)

    print("DOWNLOAD COMPLETED")

    print("=" * 80)

    print(
        "Number of symbols:",
        number_of_symbols
    )

    print(
        "Total rows:",
        total_rows
    )


# =====================================
# 3. 启动程序
# =====================================

if __name__ == "__main__":

    main()