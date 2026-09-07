from pathlib import Path


# =====================================
# 1. 项目文件夹路径
# =====================================

BASE_DIR = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

DATA_DIR = (
    BASE_DIR
    / "data"
)

REPORT_DIR = (
    BASE_DIR
    / "reports"
)

CHART_DIR = (
    REPORT_DIR
    / "charts"
)

RESULT_DIR = (
    REPORT_DIR
    / "results"
)

DATABASE_PATH = (
    DATA_DIR
    / "quant_data.db"
)


# =====================================
# 2. 回测时间设置
# =====================================

START_DATE = "2020-01-01"

END_DATE = None

TRADING_DAYS_PER_YEAR = 252


# =====================================
# 3. 初始资金
# =====================================

INITIAL_CAPITAL = 100000.0


# =====================================
# 4. 模拟交易成本
# =====================================

BUY_COMMISSION_RATE = 0.0003

SELL_COMMISSION_RATE = 0.0003

STAMP_DUTY_RATE = 0.0005

SLIPPAGE_RATE = 0.0002


# =====================================
# 5. 组合设置
# =====================================

TOP_N_STOCKS = 5

MAX_STOCK_WEIGHT = 0.20

MINIMUM_HISTORY_DAYS = 120

REBALANCE_FREQUENCY = "ME"


# =====================================
# 6. 基准指数
# =====================================

BENCHMARK_SYMBOL = "000300.SS"

BENCHMARK_NAME = "CSI 300"


# =====================================
# 7. 第一版股票池
# =====================================

STOCK_POOL = {
    "600519.SS": "Kweichow Moutai",
    "601318.SS": "Ping An Insurance",
    "600036.SS": "China Merchants Bank",
    "600276.SS": "Hengrui Medicine",
    "600900.SS": "Yangtze Power",
    "600030.SS": "CITIC Securities",
    "000001.SZ": "Ping An Bank",
    "000333.SZ": "Midea Group",
    "000651.SZ": "Gree Electric",
    "000858.SZ": "Wuliangye",
    "300750.SZ": "CATL",
    "002594.SZ": "BYD"
}


# =====================================
# 8. 因子设置
# =====================================

MOMENTUM_SHORT_DAYS = 20

MOMENTUM_LONG_DAYS = 60

REVERSAL_DAYS = 5

VOLATILITY_DAYS = 20

SHORT_MA_DAYS = 20

LONG_MA_DAYS = 60

VOLUME_DAYS = 20


# =====================================
# 9. 自动创建需要的文件夹
# =====================================

DATA_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CHART_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)