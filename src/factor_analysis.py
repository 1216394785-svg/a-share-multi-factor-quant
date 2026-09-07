import sqlite3

import numpy as np
import pandas as pd

from config.settings import (
    DATABASE_PATH,
    RESULT_DIR,
    STOCK_POOL
)


# =====================================
# 1. 设置因子权重
# =====================================

FACTOR_WEIGHTS = {
    "momentum_20_score": 0.20,
    "momentum_60_score": 0.20,
    "reversal_5_score": 0.15,
    "volatility_20_score": 0.20,
    "trend_score_normalized": 0.15,
    "volume_ratio_score": 0.10
}


# =====================================
# 2. 从数据库读取原始因子
# =====================================

def load_factor_data():

    connection = sqlite3.connect(
        DATABASE_PATH
    )


    query = """
    SELECT
        symbol,
        date,
        momentum_20,
        momentum_60,
        reversal_5,
        volatility_20,
        trend_score,
        volume_ratio
    FROM factor_values
    ORDER BY date, symbol;
    """


    factor_data = pd.read_sql_query(
        query,
        connection
    )


    connection.close()


    if factor_data.empty:

        raise ValueError(
            "No factor data found. "
            "Please run python -m src.factors first."
        )


    factor_data["date"] = (
        pd.to_datetime(
            factor_data["date"]
        )
    )


    return factor_data


# =====================================
# 3. 缩尾处理函数
# =====================================

def winsorize_by_date(
    factor_data,
    column_name
):

    cleaned_data = (
        factor_data
        .groupby("date")[
            column_name
        ]
        .transform(
            lambda values: values.clip(
                lower=values.quantile(0.05),
                upper=values.quantile(0.95)
            )
        )
    )


    return cleaned_data


# =====================================
# 4. Z-score标准化函数
# =====================================

def zscore_by_date(
    factor_data,
    column_name
):

    def calculate_zscore(values):

        standard_deviation = (
            values.std(
                ddof=0
            )
        )


        if (
            standard_deviation == 0
            or pd.isna(
                standard_deviation
            )
        ):

            return pd.Series(
                np.zeros(
                    len(values)
                ),
                index=values.index
            )


        return (
            values
            - values.mean()
        ) / standard_deviation


    standardized_data = (
        factor_data
        .groupby("date")[
            column_name
        ]
        .transform(
            calculate_zscore
        )
    )


    return standardized_data


# =====================================
# 5. 计算所有标准化因子
# =====================================

def calculate_factor_scores(
    factor_data
):

    factor_data = factor_data.copy()


    raw_factor_columns = [
        "momentum_20",
        "momentum_60",
        "reversal_5",
        "volatility_20",
        "trend_score",
        "volume_ratio"
    ]


    # ---------------------------------
    # 先对原始因子进行缩尾处理
    # ---------------------------------

    for column_name in (
        raw_factor_columns
    ):

        clean_column = (
            column_name
            + "_clean"
        )


        factor_data[clean_column] = (
            winsorize_by_date(
                factor_data,
                column_name
            )
        )


    # ---------------------------------
    # 动量因子：越高越好
    # ---------------------------------

    factor_data[
        "momentum_20_score"
    ] = zscore_by_date(
        factor_data,
        "momentum_20_clean"
    )


    factor_data[
        "momentum_60_score"
    ] = zscore_by_date(
        factor_data,
        "momentum_60_clean"
    )


    # ---------------------------------
    # 反转因子：已经设置为越高越好
    # ---------------------------------

    factor_data[
        "reversal_5_score"
    ] = zscore_by_date(
        factor_data,
        "reversal_5_clean"
    )


    # ---------------------------------
    # 波动率因子：风险越低越好
    # 所以标准化以后加负号
    # ---------------------------------

    factor_data[
        "volatility_20_score"
    ] = -zscore_by_date(
        factor_data,
        "volatility_20_clean"
    )


    # ---------------------------------
    # 趋势因子：越高越好
    # ---------------------------------

    factor_data[
        "trend_score_normalized"
    ] = zscore_by_date(
        factor_data,
        "trend_score_clean"
    )


    # ---------------------------------
    # 成交量因子：越高越好
    # ---------------------------------

    factor_data[
        "volume_ratio_score"
    ] = zscore_by_date(
        factor_data,
        "volume_ratio_clean"
    )


    return factor_data


# =====================================
# 6. 计算综合因子分数
# =====================================

def calculate_composite_score(
    factor_data
):

    factor_data = factor_data.copy()


    factor_data[
        "composite_score"
    ] = 0.0


    for score_name, weight in (
        FACTOR_WEIGHTS.items()
    ):

        factor_data[
            "composite_score"
        ] += (
            factor_data[
                score_name
            ]
            * weight
        )


    # 每个交易日分别排名
    factor_data[
        "factor_rank"
    ] = (
        factor_data
        .groupby("date")[
            "composite_score"
        ]
        .rank(
            ascending=False,
            method="first"
        )
    )


    factor_data[
        "factor_rank"
    ] = (
        factor_data[
            "factor_rank"
        ]
        .astype(int)
    )


    return factor_data


# =====================================
# 7. 保存评分结果
# =====================================

def save_factor_scores(
    factor_data
):

    save_data = factor_data.copy()


    save_data["date"] = (
        save_data["date"]
        .dt.strftime("%Y-%m-%d")
    )


    output_columns = [
        "symbol",
        "date",
        "momentum_20",
        "momentum_60",
        "reversal_5",
        "volatility_20",
        "trend_score",
        "volume_ratio",
        "momentum_20_score",
        "momentum_60_score",
        "reversal_5_score",
        "volatility_20_score",
        "trend_score_normalized",
        "volume_ratio_score",
        "composite_score",
        "factor_rank"
    ]


    connection = sqlite3.connect(
        DATABASE_PATH
    )


    save_data[
        output_columns
    ].to_sql(
        "factor_scores",
        connection,
        if_exists="replace",
        index=False
    )


    update_records = list(
        save_data[
            [
                "composite_score",
                "symbol",
                "date"
            ]
        ].itertuples(
            index=False,
            name=None
        )
    )


    cursor = connection.cursor()


    cursor.executemany(
        """
        UPDATE factor_values
        SET composite_score = ?
        WHERE symbol = ?
        AND date = ?;
        """,
        update_records
    )


    connection.commit()

    connection.close()


    print(
        f"Saved {len(save_data)} "
        f"factor score rows."
    )


# =====================================
# 8. 显示最新选股排名
# =====================================

def show_latest_ranking(
    factor_data
):

    latest_date = (
        factor_data["date"]
        .max()
    )


    latest_ranking = factor_data[
        factor_data["date"]
        == latest_date
    ].copy()


    latest_ranking[
        "stock_name"
    ] = (
        latest_ranking["symbol"]
        .map(STOCK_POOL)
    )


    latest_ranking = (
        latest_ranking
        .sort_values(
            "factor_rank"
        )
    )


    display_columns = [
        "factor_rank",
        "symbol",
        "stock_name",
        "momentum_20_score",
        "momentum_60_score",
        "reversal_5_score",
        "volatility_20_score",
        "trend_score_normalized",
        "volume_ratio_score",
        "composite_score"
    ]


    latest_ranking[
        display_columns
    ].to_csv(
        RESULT_DIR
        / "latest_factor_ranking.csv",
        index=False,
        encoding="utf-8-sig"
    )


    print("\n" + "=" * 100)

    print(
        "LATEST FACTOR RANKING"
    )

    print("=" * 100)

    print(
        "Ranking date:",
        latest_date.date()
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
        "{:,.4f}".format
    )


    print(
        latest_ranking[
            display_columns
        ].to_string(
            index=False
        )
    )


    return latest_ranking


# =====================================
# 9. 主程序
# =====================================

def main():

    print("\nCalculating factor scores...")


    factor_data = (
        load_factor_data()
    )


    factor_data = (
        calculate_factor_scores(
            factor_data
        )
    )


    factor_data = (
        calculate_composite_score(
            factor_data
        )
    )


    save_factor_scores(
        factor_data
    )


    show_latest_ranking(
        factor_data
    )


    print(
        "\nFactor analysis completed."
    )


# =====================================
# 10. 启动程序
# =====================================

if __name__ == "__main__":
    main()