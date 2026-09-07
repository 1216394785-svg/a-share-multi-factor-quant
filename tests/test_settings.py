from config.settings import (
    DATA_DIR,
    CHART_DIR,
    RESULT_DIR,
    DATABASE_PATH,
    INITIAL_CAPITAL,
    TOP_N_STOCKS,
    STOCK_POOL
)


# =====================================
# 测试1：检查文件夹是否存在
# =====================================

def test_project_folders_exist():

    assert DATA_DIR.exists()

    assert CHART_DIR.exists()

    assert RESULT_DIR.exists()


# =====================================
# 测试2：检查初始资金
# =====================================

def test_initial_capital():

    assert INITIAL_CAPITAL > 0

    assert INITIAL_CAPITAL == 100000.0


# =====================================
# 测试3：检查股票池
# =====================================

def test_stock_pool():

    assert len(STOCK_POOL) > 0

    assert len(STOCK_POOL) >= TOP_N_STOCKS


# =====================================
# 测试4：检查数据库路径
# =====================================

def test_database_path():

    assert (
        DATABASE_PATH.name
        == "quant_data.db"
    )

    assert (
        DATABASE_PATH.parent
        == DATA_DIR
    )