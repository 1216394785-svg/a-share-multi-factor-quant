import sqlite3

from config.settings import (
    DATABASE_PATH
)


# =====================================
# 1. 创建数据库连接
# =====================================

def get_connection():

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    return connection


# =====================================
# 2. 创建项目需要的数据表
# =====================================

def initialize_database():

    connection = get_connection()

    cursor = connection.cursor()


    # ---------------------------------
    # 股票每日行情表
    # ---------------------------------

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS daily_prices (
            symbol TEXT NOT NULL,
            date TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            PRIMARY KEY (symbol, date)
        );
        """
    )


    # ---------------------------------
    # 因子结果表
    # ---------------------------------

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS factor_values (
            symbol TEXT NOT NULL,
            date TEXT NOT NULL,
            momentum_20 REAL,
            momentum_60 REAL,
            reversal_5 REAL,
            volatility_20 REAL,
            trend_score REAL,
            volume_ratio REAL,
            composite_score REAL,
            PRIMARY KEY (symbol, date)
        );
        """
    )


    # ---------------------------------
    # 每次调仓的持仓表
    # ---------------------------------

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS portfolio_holdings (
            date TEXT NOT NULL,
            symbol TEXT NOT NULL,
            weight REAL,
            price REAL,
            shares REAL,
            PRIMARY KEY (date, symbol)
        );
        """
    )


    # ---------------------------------
    # 每日回测结果表
    # ---------------------------------

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS backtest_results (
            date TEXT PRIMARY KEY,
            portfolio_value REAL,
            benchmark_value REAL,
            cash REAL,
            daily_return REAL,
            benchmark_return REAL,
            drawdown REAL
        );
        """
    )


    # 保存数据库变化
    connection.commit()


    # 关闭数据库
    connection.close()


# =====================================
# 3. 查询数据库中有哪些表
# =====================================

def list_tables():

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
        ORDER BY name;
        """
    )

    results = cursor.fetchall()

    connection.close()


    table_names = [
        row[0]
        for row in results
    ]

    return table_names


# =====================================
# 4. 直接运行本文件时执行
# =====================================

if __name__ == "__main__":

    initialize_database()

    tables = list_tables()

    print(
        "Database created successfully."
    )

    print(
        "Database path:",
        DATABASE_PATH
    )

    print(
        "Tables:"
    )

    for table in tables:

        print(
            "-",
            table
        )