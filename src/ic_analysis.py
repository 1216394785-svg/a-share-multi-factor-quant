import sqlite3

import numpy as np
import pandas as pd

from config.settings import (
    DATABASE_PATH,
    RESULT_DIR
)


# =====================================
# 1. 设置未来收益期限
# =====================================

FUTURE_RETURN_DAYS = 20

MINIMUM_STOCKS_PER_DATE = 6


# =====================================
# 2. 设置需要检验的因子
# =====================================

FACTOR_COLUMNS = {
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
    ),

    "composite_score": (
        "Composite Factor"
    )
}


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
        composite_score
    FROM factor_scores
    ORDER BY symbol, date;
    """


    factor_data = pd.read_sql_query(
        query,
        connection
    )


    connection.close()


    if factor_data.empty:

        raise ValueError(
            "No factor scores found."
        )


    factor_data["date"] = (
        pd.to_datetime(
            factor_data["date"]
        )
    )


    return factor_data


# =====================================
# 4. 读取收盘价
# =====================================

def load_stock_prices():

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
# 5. 读取月末信号日期
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


    signal_dates = pd.read_sql_query(
        query,
        connection
    )


    connection.close()


    if signal_dates.empty:

        raise ValueError(
            "No rebalance signal dates found."
        )


    signal_dates["signal_date"] = (
        pd.to_datetime(
            signal_dates["signal_date"]
        )
    )


    return set(
        signal_dates[
            "signal_date"
        ]
    )


# =====================================
# 6. 计算未来20日收益
# =====================================

def calculate_future_returns(
    price_data
):

    price_data = price_data.copy()


    price_data = (
        price_data
        .sort_values(
            [
                "symbol",
                "date"
            ]
        )
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


    return price_data[
        [
            "symbol",
            "date",
            "future_return"
        ]
    ]


# =====================================
# 7. 合并因子与未来收益
# =====================================

def prepare_ic_data(
    factor_data,
    future_returns,
    signal_dates
):

    merged_data = pd.merge(
        factor_data,
        future_returns,
        on=[
            "symbol",
            "date"
        ],
        how="inner"
    )


    merged_data = merged_data[
        merged_data["date"].isin(
            signal_dates
        )
    ].copy()


    merged_data = merged_data.dropna(
        subset=[
            "future_return"
        ]
    )


    return merged_data


# =====================================
# 8. 计算每个月的Rank IC
# =====================================

def calculate_monthly_ic(
    ic_data
):

    all_ic_results = []


    for signal_date, daily_data in (
        ic_data.groupby("date")
    ):

        daily_result = {
            "date": signal_date
        }


        for factor_column in (
            FACTOR_COLUMNS.keys()
        ):

            valid_data = daily_data[
                [
                    factor_column,
                    "future_return"
                ]
            ].dropna()


            if (
                len(valid_data)
                < MINIMUM_STOCKS_PER_DATE
            ):

                daily_result[
                    factor_column
                ] = np.nan

                continue


            if (
                valid_data[
                    factor_column
                ].nunique()
                <= 1
            ):

                daily_result[
                    factor_column
                ] = np.nan

                continue


            rank_ic = (
                valid_data[
                    factor_column
                ]
                .corr(
                    valid_data[
                        "future_return"
                    ],
                    method="spearman"
                )
            )


            daily_result[
                factor_column
            ] = rank_ic


        all_ic_results.append(
            daily_result
        )


    monthly_ic = pd.DataFrame(
        all_ic_results
    )


    monthly_ic = (
        monthly_ic
        .sort_values("date")
        .reset_index(drop=True)
    )


    return monthly_ic


# =====================================
# 9. 计算IC汇总指标
# =====================================

def calculate_ic_summary(
    monthly_ic
):

    summary_results = []


    for factor_column, factor_name in (
        FACTOR_COLUMNS.items()
    ):

        ic_series = (
            monthly_ic[
                factor_column
            ]
            .dropna()
        )


        number_of_periods = len(
            ic_series
        )


        if number_of_periods == 0:

            continue


        mean_ic = (
            ic_series.mean()
        )


        ic_standard_deviation = (
            ic_series.std()
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
                        number_of_periods
                    )
                )
            )


        positive_ic_rate = (
            (
                ic_series > 0
            ).mean()
        )


        summary_results.append(
            {
                "factor_column": (
                    factor_column
                ),

                "factor_name": (
                    factor_name
                ),

                "number_of_periods": (
                    number_of_periods
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
                    positive_ic_rate * 100
                ),

                "t_statistic": (
                    t_statistic
                )
            }
        )


    ic_summary = pd.DataFrame(
        summary_results
    )


    ic_summary = (
        ic_summary
        .sort_values(
            by="mean_ic",
            ascending=False
        )
        .reset_index(drop=True)
    )


    return ic_summary


# =====================================
# 10. 保存IC结果
# =====================================

def save_ic_results(
    monthly_ic,
    ic_summary
):

    save_monthly_ic = (
        monthly_ic.copy()
    )


    save_monthly_ic["date"] = (
        save_monthly_ic["date"]
        .dt.strftime("%Y-%m-%d")
    )


    connection = sqlite3.connect(
        DATABASE_PATH
    )


    save_monthly_ic.to_sql(
        "monthly_factor_ic",
        connection,
        if_exists="replace",
        index=False
    )


    ic_summary.to_sql(
        "factor_ic_summary",
        connection,
        if_exists="replace",
        index=False
    )


    connection.close()


    save_monthly_ic.to_csv(
        RESULT_DIR
        / "monthly_factor_ic.csv",
        index=False,
        encoding="utf-8-sig"
    )


    ic_summary.to_csv(
        RESULT_DIR
        / "factor_ic_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )


# =====================================
# 11. 显示分析结果
# =====================================

def show_ic_results(
    monthly_ic,
    ic_summary
):

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


    print("\n" + "=" * 100)

    print("FACTOR IC SUMMARY")

    print("=" * 100)


    print(
        ic_summary.to_string(
            index=False
        )
    )


    print("\n" + "=" * 100)

    print("LATEST 10 MONTHLY IC RESULTS")

    print("=" * 100)


    print(
        monthly_ic.tail(10).to_string(
            index=False
        )
    )


    best_factor = (
        ic_summary.iloc[0]
    )


    print("\n" + "=" * 100)

    print("SIMPLE CONCLUSION")

    print("=" * 100)


    print(
        "Best factor:",
        best_factor[
            "factor_name"
        ]
    )


    print(
        f"Mean IC: "
        f"{best_factor['mean_ic']:.4f}"
    )


    print(
        f"Positive IC rate: "
        f"{best_factor['positive_ic_rate_percent']:.2f}%"
    )


# =====================================
# 12. 主程序
# =====================================

def main():

    print(
        "\nCalculating factor IC..."
    )


    factor_data = (
        load_factor_scores()
    )


    price_data = (
        load_stock_prices()
    )


    signal_dates = (
        load_signal_dates()
    )


    future_returns = (
        calculate_future_returns(
            price_data
        )
    )


    ic_data = (
        prepare_ic_data(
            factor_data,
            future_returns,
            signal_dates
        )
    )


    monthly_ic = (
        calculate_monthly_ic(
            ic_data
        )
    )


    ic_summary = (
        calculate_ic_summary(
            monthly_ic
        )
    )


    save_ic_results(
        monthly_ic,
        ic_summary
    )


    show_ic_results(
        monthly_ic,
        ic_summary
    )


    print(
        "\nIC analysis completed."
    )


# =====================================
# 13. 启动程序
# =====================================

if __name__ == "__main__":

    main()