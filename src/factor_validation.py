import sqlite3

import numpy as np
import pandas as pd

from config.settings import (
    DATABASE_PATH,
    RESULT_DIR
)


# =====================================
# 1. 基本设置
# =====================================

TRAINING_RATIO = 0.70

FUTURE_RETURN_DAYS = 20

MINIMUM_STOCKS_PER_DATE = 6


FACTOR_NAMES = {
    "momentum_20_score": (
        "20-Day Momentum"
    ),

    "momentum_60_score": (
        "60-Day Momentum"
    ),

    "reversal_5_score": (
        "5-Day Reversal"
    ),

    "volatility_20_score": (
        "Low Volatility"
    ),

    "trend_score_normalized": (
        "MA Trend"
    ),

    "volume_ratio_score": (
        "Volume Ratio"
    )
}


# =====================================
# 2. 读取因子评分
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
        volume_ratio_score
    FROM factor_scores
    ORDER BY symbol, date;
    """


    factor_data = pd.read_sql_query(
        query,
        connection
    )


    connection.close()


    factor_data["date"] = (
        pd.to_datetime(
            factor_data["date"]
        )
    )


    return factor_data


# =====================================
# 3. 读取股票价格
# =====================================

def load_price_data():

    connection = sqlite3.connect(
        DATABASE_PATH
    )


    query = """
    SELECT
        symbol,
        date,
        close
    FROM daily_prices
    ORDER BY symbol, date;
    """


    price_data = pd.read_sql_query(
        query,
        connection
    )


    connection.close()


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


    return price_data


# =====================================
# 4. 读取月度信号日期
# =====================================

def load_signal_dates():

    connection = sqlite3.connect(
        DATABASE_PATH
    )


    query = """
    SELECT DISTINCT signal_date
    FROM rebalance_orders
    ORDER BY signal_date;
    """


    signal_data = pd.read_sql_query(
        query,
        connection
    )


    connection.close()


    signal_data["signal_date"] = (
        pd.to_datetime(
            signal_data["signal_date"]
        )
    )


    return set(
        signal_data["signal_date"]
    )


# =====================================
# 5. 合并因子和未来收益
# =====================================

def prepare_validation_data(
    factor_data,
    price_data,
    signal_dates
):

    price_data = (
        price_data
        .sort_values(
            [
                "symbol",
                "date"
            ]
        )
        .copy()
    )


    future_close = (
        price_data
        .groupby("symbol")[
            "close"
        ]
        .shift(
            -FUTURE_RETURN_DAYS
        )
    )


    price_data[
        "future_return"
    ] = (
        future_close
        / price_data["close"]
        - 1
    )


    future_returns = price_data[
        [
            "symbol",
            "date",
            "future_return"
        ]
    ]


    validation_data = pd.merge(
        factor_data,
        future_returns,
        on=[
            "symbol",
            "date"
        ],
        how="inner"
    )


    validation_data = validation_data[
        validation_data["date"].isin(
            signal_dates
        )
    ].copy()


    validation_data = (
        validation_data
        .dropna(
            subset=[
                "future_return"
            ]
        )
    )


    return validation_data


# =====================================
# 6. 计算一个日期的Rank IC
# =====================================

def calculate_rank_ic(
    daily_data,
    score_column
):

    valid_data = daily_data[
        [
            score_column,
            "future_return"
        ]
    ].dropna()


    if (
        len(valid_data)
        < MINIMUM_STOCKS_PER_DATE
    ):

        return np.nan


    if (
        valid_data[
            score_column
        ].nunique()
        <= 1
    ):

        return np.nan


    rank_ic = (
        valid_data[
            score_column
        ]
        .corr(
            valid_data[
                "future_return"
            ],
            method="spearman"
        )
    )


    return rank_ic


# =====================================
# 7. 计算某个因子的月度IC
# =====================================

def calculate_factor_ic_series(
    validation_data,
    score_column,
    selected_dates
):

    ic_records = []


    selected_data = validation_data[
        validation_data["date"].isin(
            selected_dates
        )
    ]


    for signal_date, daily_data in (
        selected_data.groupby("date")
    ):

        rank_ic = calculate_rank_ic(
            daily_data,
            score_column
        )


        ic_records.append(
            {
                "date": signal_date,
                "rank_ic": rank_ic
            }
        )


    ic_series = pd.DataFrame(
        ic_records
    )


    return ic_series


# =====================================
# 8. 根据训练集确定方向和权重
# =====================================

def train_factor_weights(
    validation_data,
    training_dates,
    testing_dates
):

    weight_records = []


    for factor_column, factor_name in (
        FACTOR_NAMES.items()
    ):

        training_ic = (
            calculate_factor_ic_series(
                validation_data,
                factor_column,
                training_dates
            )
        )


        testing_ic = (
            calculate_factor_ic_series(
                validation_data,
                factor_column,
                testing_dates
            )
        )


        train_mean_ic = (
            training_ic[
                "rank_ic"
            ].mean()
        )


        test_mean_ic = (
            testing_ic[
                "rank_ic"
            ].mean()
        )


        train_positive_rate = (
            (
                training_ic[
                    "rank_ic"
                ] > 0
            ).mean()
        )


        test_positive_rate = (
            (
                testing_ic[
                    "rank_ic"
                ] > 0
            ).mean()
        )


        if train_mean_ic >= 0:

            direction = 1

        else:

            direction = -1


        raw_weight = abs(
            train_mean_ic
        )


        weight_records.append(
            {
                "factor_column": (
                    factor_column
                ),

                "factor_name": (
                    factor_name
                ),

                "train_mean_ic": (
                    train_mean_ic
                ),

                "test_mean_ic": (
                    test_mean_ic
                ),

                "direction": (
                    direction
                ),

                "raw_weight": (
                    raw_weight
                ),

                "train_positive_rate_percent": (
                    train_positive_rate
                    * 100
                ),

                "test_positive_rate_percent": (
                    test_positive_rate
                    * 100
                )
            }
        )


    factor_weights = pd.DataFrame(
        weight_records
    )


    total_raw_weight = (
        factor_weights[
            "raw_weight"
        ].sum()
    )


    if total_raw_weight == 0:

        factor_weights[
            "trained_weight"
        ] = (
            1
            / len(factor_weights)
        )

    else:

        factor_weights[
            "trained_weight"
        ] = (
            factor_weights[
                "raw_weight"
            ]
            / total_raw_weight
        )


    factor_weights[
        "same_direction_in_test"
    ] = (
        np.sign(
            factor_weights[
                "train_mean_ic"
            ]
        )
        == np.sign(
            factor_weights[
                "test_mean_ic"
            ]
        )
    )


    return factor_weights


# =====================================
# 9. 建立训练后的综合因子
# =====================================

def create_trained_composite_score(
    validation_data,
    factor_weights
):

    validation_data = (
        validation_data.copy()
    )


    validation_data[
        "trained_composite_score"
    ] = 0.0


    for _, factor in (
        factor_weights.iterrows()
    ):

        factor_column = (
            factor["factor_column"]
        )


        direction = (
            factor["direction"]
        )


        weight = (
            factor["trained_weight"]
        )


        validation_data[
            "trained_composite_score"
        ] += (
            validation_data[
                factor_column
            ]
            * direction
            * weight
        )


    return validation_data


# =====================================
# 10. 检验综合因子
# =====================================

def evaluate_composite_factor(
    validation_data,
    training_dates,
    testing_dates
):

    training_ic = (
        calculate_factor_ic_series(
            validation_data,
            "trained_composite_score",
            training_dates
        )
    )


    testing_ic = (
        calculate_factor_ic_series(
            validation_data,
            "trained_composite_score",
            testing_dates
        )
    )


    training_ic[
        "period"
    ] = "Training"


    testing_ic[
        "period"
    ] = "Testing"


    combined_ic = pd.concat(
        [
            training_ic,
            testing_ic
        ],
        ignore_index=True
    )


    summary_records = []


    for period_name, period_data in (
        combined_ic.groupby("period")
    ):

        ic_values = (
            period_data[
                "rank_ic"
            ]
            .dropna()
        )


        mean_ic = (
            ic_values.mean()
        )


        ic_standard_deviation = (
            ic_values.std()
        )


        if (
            ic_standard_deviation == 0
            or pd.isna(
                ic_standard_deviation
            )
        ):

            ic_ir = 0

            t_statistic = 0

        else:

            ic_ir = (
                mean_ic
                / ic_standard_deviation
            )


            t_statistic = (
                mean_ic
                / (
                    ic_standard_deviation
                    / np.sqrt(
                        len(ic_values)
                    )
                )
            )


        positive_rate = (
            (
                ic_values > 0
            ).mean()
        )


        summary_records.append(
            {
                "period": (
                    period_name
                ),

                "number_of_months": (
                    len(ic_values)
                ),

                "mean_ic": (
                    mean_ic
                ),

                "ic_standard_deviation": (
                    ic_standard_deviation
                ),

                "ic_ir": (
                    ic_ir
                ),

                "positive_ic_rate_percent": (
                    positive_rate * 100
                ),

                "t_statistic": (
                    t_statistic
                )
            }
        )


    composite_summary = pd.DataFrame(
        summary_records
    )


    return (
        combined_ic,
        composite_summary
    )


# =====================================
# 11. 保存验证结果
# =====================================

def save_validation_results(
    factor_weights,
    combined_ic,
    composite_summary
):

    save_ic = combined_ic.copy()


    save_ic["date"] = (
        save_ic["date"]
        .dt.strftime("%Y-%m-%d")
    )


    connection = sqlite3.connect(
        DATABASE_PATH
    )


    factor_weights.to_sql(
        "trained_factor_weights",
        connection,
        if_exists="replace",
        index=False
    )


    save_ic.to_sql(
        "trained_composite_ic",
        connection,
        if_exists="replace",
        index=False
    )


    composite_summary.to_sql(
        "factor_validation_summary",
        connection,
        if_exists="replace",
        index=False
    )


    connection.close()


    factor_weights.to_csv(
        RESULT_DIR
        / "trained_factor_weights.csv",
        index=False,
        encoding="utf-8-sig"
    )


    composite_summary.to_csv(
        RESULT_DIR
        / "factor_validation_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )


# =====================================
# 12. 主程序
# =====================================

def main():

    print(
        "\nRunning factor validation..."
    )


    factor_data = (
        load_factor_scores()
    )


    price_data = (
        load_price_data()
    )


    signal_dates = (
        load_signal_dates()
    )


    validation_data = (
        prepare_validation_data(
            factor_data,
            price_data,
            signal_dates
        )
    )


    available_dates = sorted(
        validation_data[
            "date"
        ].unique()
    )


    split_position = int(
        len(available_dates)
        * TRAINING_RATIO
    )


    training_dates = (
        available_dates[
            :split_position
        ]
    )


    testing_dates = (
        available_dates[
            split_position:
        ]
    )


    factor_weights = (
        train_factor_weights(
            validation_data,
            training_dates,
            testing_dates
        )
    )


    validation_data = (
        create_trained_composite_score(
            validation_data,
            factor_weights
        )
    )


    (
        combined_ic,
        composite_summary
    ) = evaluate_composite_factor(
        validation_data,
        training_dates,
        testing_dates
    )


    save_validation_results(
        factor_weights,
        combined_ic,
        composite_summary
    )


    pd.set_option(
        "display.max_columns",
        None
    )

    pd.set_option(
        "display.width",
        240
    )

    pd.set_option(
        "display.float_format",
        "{:,.4f}".format
    )


    print("\n" + "=" * 110)

    print("TRAINED FACTOR WEIGHTS")

    print("=" * 110)


    print(
        factor_weights.to_string(
            index=False
        )
    )


    print("\n" + "=" * 110)

    print("TRAINED COMPOSITE FACTOR VALIDATION")

    print("=" * 110)


    print(
        composite_summary.to_string(
            index=False
        )
    )


    print("\nTraining period:")

    print(
        pd.Timestamp(
            training_dates[0]
        ).date(),
        "to",
        pd.Timestamp(
            training_dates[-1]
        ).date()
    )


    print("\nTesting period:")

    print(
        pd.Timestamp(
            testing_dates[0]
        ).date(),
        "to",
        pd.Timestamp(
            testing_dates[-1]
        ).date()
    )


    print(
        "\nFactor validation completed."
    )


# =====================================
# 13. 启动程序
# =====================================

if __name__ == "__main__":

    main()