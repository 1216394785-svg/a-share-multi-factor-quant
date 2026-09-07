import sqlite3
import webbrowser

import pandas as pd

from config.settings import (
    DATABASE_PATH,
    REPORT_DIR,
    STOCK_POOL
)


# =====================================
# 1. 从数据库读取数据表
# =====================================

def read_table(
    table_name
):

    connection = sqlite3.connect(
        DATABASE_PATH
    )


    query = (
        f"SELECT * FROM {table_name};"
    )


    data = pd.read_sql_query(
        query,
        connection
    )


    connection.close()


    if data.empty:

        raise ValueError(
            f"Table {table_name} is empty."
        )


    return data


# =====================================
# 2. 读取最新因子排名
# =====================================

def load_latest_factor_ranking():

    connection = sqlite3.connect(
        DATABASE_PATH
    )


    latest_date_query = """
    SELECT MAX(date) AS latest_date
    FROM factor_scores;
    """


    latest_date_result = (
        pd.read_sql_query(
            latest_date_query,
            connection
        )
    )


    latest_date = (
        latest_date_result.loc[
            0,
            "latest_date"
        ]
    )


    ranking_query = """
    SELECT
        factor_rank,
        symbol,
        composite_score
    FROM factor_scores
    WHERE date = ?
    ORDER BY factor_rank
    LIMIT 5;
    """


    ranking = pd.read_sql_query(
        ranking_query,
        connection,
        params=(
            latest_date,
        )
    )


    connection.close()


    ranking[
        "stock_name"
    ] = (
        ranking["symbol"]
        .map(STOCK_POOL)
    )


    ranking = ranking[
        [
            "factor_rank",
            "symbol",
            "stock_name",
            "composite_score"
        ]
    ]


    return (
        latest_date,
        ranking
    )


# =====================================
# 3. 把DataFrame转换成HTML表格
# =====================================

def dataframe_to_html(
    data
):

    formatted_data = data.copy()


    numeric_columns = (
        formatted_data
        .select_dtypes(
            include="number"
        )
        .columns
    )


    for column in numeric_columns:

        formatted_data[column] = (
            formatted_data[column]
            .round(2)
        )


    html_table = (
        formatted_data.to_html(
            index=False,
            border=0,
            classes="data-table",
            justify="center",
            na_rep="N/A"
        )
    )


    return html_table


# =====================================
# 4. 生成研究报告
# =====================================

def generate_report():

    oos_performance = read_table(
        "oos_performance_summary"
    )


    factor_validation = read_table(
        "factor_validation_summary"
    )


    risk_summary = read_table(
        "oos_risk_summary"
    )


    turnover_summary = read_table(
        "oos_turnover_summary"
    )


    trained_weights = read_table(
        "trained_factor_weights"
    )


    (
        latest_ranking_date,
        latest_ranking
    ) = load_latest_factor_ranking()


    # ---------------------------------
    # 提取核心结果
    # ---------------------------------

    trained_result = (
        oos_performance[
            oos_performance["method"]
            == "Trained Factor"
        ]
        .iloc[0]
    )


    original_result = (
        oos_performance[
            oos_performance["method"]
            == "Original Factor"
        ]
        .iloc[0]
    )


    benchmark_result = (
        oos_performance[
            oos_performance["method"]
            == "CSI 300 Benchmark"
        ]
        .iloc[0]
    )


    trained_return = float(
        trained_result[
            "total_return_percent"
        ]
    )


    original_return = float(
        original_result[
            "total_return_percent"
        ]
    )


    benchmark_return = float(
        benchmark_result[
            "total_return_percent"
        ]
    )


    trained_sharpe = float(
        trained_result[
            "sharpe_ratio"
        ]
    )


    trained_drawdown = float(
        trained_result[
            "maximum_drawdown_percent"
        ]
    )


    excess_return = (
        trained_return
        - benchmark_return
    )


    improvement = (
        trained_return
        - original_return
    )


    # ---------------------------------
    # 整理绩效表
    # ---------------------------------

    performance_display = (
        oos_performance[
            [
                "method",
                "final_value",
                "total_return_percent",
                "annual_return_percent",
                "annual_volatility_percent",
                "sharpe_ratio",
                "maximum_drawdown_percent",
                "total_cost_percent"
            ]
        ]
        .copy()
    )


    performance_display.columns = [
        "Method",
        "Final Value",
        "Total Return (%)",
        "Annual Return (%)",
        "Annual Volatility (%)",
        "Sharpe Ratio",
        "Maximum Drawdown (%)",
        "Cost (%)"
    ]


    # ---------------------------------
    # 整理因子验证表
    # ---------------------------------

    validation_display = (
        factor_validation[
            [
                "period",
                "number_of_months",
                "mean_ic",
                "ic_standard_deviation",
                "ic_ir",
                "positive_ic_rate_percent",
                "t_statistic"
            ]
        ]
        .copy()
    )


    validation_display.columns = [
        "Period",
        "Months",
        "Mean IC",
        "IC Standard Deviation",
        "IC IR",
        "Positive IC Rate (%)",
        "t-Statistic"
    ]


    # ---------------------------------
    # 整理因子权重表
    # ---------------------------------

    weight_display = (
        trained_weights[
            [
                "factor_name",
                "train_mean_ic",
                "test_mean_ic",
                "direction",
                "trained_weight",
                "same_direction_in_test"
            ]
        ]
        .copy()
    )


    weight_display[
        "trained_weight"
    ] = (
        weight_display[
            "trained_weight"
        ]
        * 100
    )


    weight_display.columns = [
        "Factor",
        "Training Mean IC",
        "Testing Mean IC",
        "Direction",
        "Weight (%)",
        "Stable Direction"
    ]


    # ---------------------------------
    # 整理风险表
    # ---------------------------------

    risk_display = (
        risk_summary[
            [
                "method",
                "var_95_percent",
                "cvar_95_percent",
                "worst_day_return_percent",
                "maximum_drawdown_percent",
                "maximum_loss_streak_days",
                "monthly_win_rate_percent"
            ]
        ]
        .copy()
    )


    risk_display.columns = [
        "Method",
        "95% VaR (%)",
        "95% CVaR (%)",
        "Worst Day (%)",
        "Maximum Drawdown (%)",
        "Maximum Loss Streak",
        "Monthly Win Rate (%)"
    ]


    # ---------------------------------
    # 整理换手率表
    # ---------------------------------

    turnover_display = (
        turnover_summary[
            [
                "method",
                "number_of_rebalances",
                "total_turnover_percent",
                "average_monthly_turnover_percent",
                "annualized_turnover_percent"
            ]
        ]
        .copy()
    )


    turnover_display.columns = [
        "Method",
        "Rebalances",
        "Total Turnover (%)",
        "Average Monthly Turnover (%)",
        "Annualised Turnover (%)"
    ]


    # ---------------------------------
    # 整理最新排名表
    # ---------------------------------

    ranking_display = (
        latest_ranking.copy()
    )


    ranking_display.columns = [
        "Rank",
        "Symbol",
        "Company",
        "Composite Score"
    ]


    # ---------------------------------
    # 自动生成结论
    # ---------------------------------

    if trained_return > benchmark_return:

        benchmark_conclusion = (
            "The trained factor model "
            "outperformed the CSI 300."
        )

    else:

        benchmark_conclusion = (
            "The trained factor model "
            "underperformed the CSI 300."
        )


    if trained_return > original_return:

        model_conclusion = (
            "Training improved performance "
            "relative to the original model."
        )

    else:

        model_conclusion = (
            "Training did not improve "
            "the original model."
        )


    # ---------------------------------
    # HTML内容
    # ---------------------------------

    html_content = f"""
<!DOCTYPE html>

<html lang="en">

<head>

    <meta charset="UTF-8">

    <meta
        name="viewport"
        content="width=device-width, initial-scale=1.0"
    >

    <title>
        A-Share Multi-Factor Research Report
    </title>

    <style>

        body {{
            margin: 0;
            padding: 0;
            background: #eef2f6;
            color: #1d2939;
            font-family:
                Arial,
                Helvetica,
                sans-serif;
            line-height: 1.6;
        }}

        .report {{
            max-width: 1180px;
            margin: 30px auto;
            background: white;
            padding: 45px 55px;
            box-shadow:
                0 6px 24px
                rgba(15, 35, 60, 0.12);
        }}

        h1 {{
            margin-bottom: 8px;
            color: #17365d;
            font-size: 34px;
        }}

        h2 {{
            margin-top: 42px;
            padding-bottom: 8px;
            border-bottom: 2px solid #d7e2ee;
            color: #1f4e79;
        }}

        h3 {{
            color: #344f6e;
        }}

        .subtitle {{
            color: #667085;
            font-size: 17px;
            margin-bottom: 28px;
        }}

        .cards {{
            display: grid;
            grid-template-columns:
                repeat(4, 1fr);
            gap: 16px;
            margin: 25px 0;
        }}

        .card {{
            background: #f4f7fb;
            border-left: 5px solid #1f4e79;
            border-radius: 6px;
            padding: 18px;
        }}

        .card-label {{
            color: #667085;
            font-size: 13px;
            text-transform: uppercase;
        }}

        .card-value {{
            color: #17365d;
            font-size: 25px;
            font-weight: bold;
            margin-top: 4px;
        }}

        .data-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 18px 0 28px 0;
            font-size: 14px;
        }}

        .data-table th {{
            background: #1f4e79;
            color: white;
            padding: 11px 8px;
            text-align: center;
        }}

        .data-table td {{
            border-bottom: 1px solid #d9e2ec;
            padding: 10px 8px;
            text-align: center;
        }}

        .data-table tr:nth-child(even) {{
            background: #f7f9fc;
        }}

        .chart {{
            width: 100%;
            margin: 20px 0;
            border: 1px solid #d7e2ee;
        }}

        .conclusion {{
            background: #f0f6ff;
            border-left: 5px solid #2f6fad;
            padding: 20px 24px;
            margin: 22px 0;
        }}

        .warning {{
            background: #fff7e6;
            border-left: 5px solid #e59f23;
            padding: 18px 22px;
        }}

        .footer {{
            margin-top: 45px;
            padding-top: 15px;
            border-top: 1px solid #d7e2ee;
            color: #667085;
            font-size: 12px;
        }}

        @media print {{

            body {{
                background: white;
            }}

            .report {{
                margin: 0;
                max-width: none;
                box-shadow: none;
            }}

        }}

    </style>

</head>

<body>

<div class="report">

    <h1>
        A-Share Multi-Factor Quantitative Research
    </h1>

    <div class="subtitle">
        Out-of-sample factor validation,
        portfolio backtesting and risk analysis
    </div>

    <div class="cards">

        <div class="card">
            <div class="card-label">
                Trained factor return
            </div>
            <div class="card-value">
                {trained_return:.2f}%
            </div>
        </div>

        <div class="card">
            <div class="card-label">
                CSI 300 return
            </div>
            <div class="card-value">
                {benchmark_return:.2f}%
            </div>
        </div>

        <div class="card">
            <div class="card-label">
                Sharpe ratio
            </div>
            <div class="card-value">
                {trained_sharpe:.2f}
            </div>
        </div>

        <div class="card">
            <div class="card-label">
                Maximum drawdown
            </div>
            <div class="card-value">
                {trained_drawdown:.2f}%
            </div>
        </div>

    </div>

    <h2>Executive Summary</h2>

    <div class="conclusion">

        <p>
            The trained multi-factor model returned
            <strong>{trained_return:.2f}%</strong>
            during the true out-of-sample period.
            The original model returned
            <strong>{original_return:.2f}%</strong>,
            while the CSI 300 returned
            <strong>{benchmark_return:.2f}%</strong>.
        </p>

        <p>
            Training improved the original strategy by
            <strong>{improvement:.2f} percentage points</strong>.
            However, the trained model produced an
            excess return of
            <strong>{excess_return:.2f} percentage points</strong>
            relative to the benchmark.
        </p>

        <p>
            {model_conclusion}
            {benchmark_conclusion}
        </p>

    </div>

    <h2>Research Methodology</h2>

    <ul>
        <li>
            Twelve selected A-share companies
            and the CSI 300 benchmark.
        </li>

        <li>
            Six price- and volume-based factors.
        </li>

        <li>
            Cross-sectional winsorisation
            and Z-score standardisation.
        </li>

        <li>
            Monthly selection of the top five stocks.
        </li>

        <li>
            Equal weighting with a 20% stock limit.
        </li>

        <li>
            Transaction commission,
            stamp duty and slippage included.
        </li>

        <li>
            Chronological 70% training
            and 30% testing split.
        </li>

        <li>
            Factor weights frozen before
            the out-of-sample backtest.
        </li>
    </ul>

    <h2>True Out-of-Sample Performance</h2>

    {dataframe_to_html(performance_display)}

    <img
        class="chart"
        src="charts/oos_backtest_report.png"
        alt="Out-of-sample backtest charts"
    >

    <h2>Factor Validation</h2>

    {dataframe_to_html(validation_display)}

    <h3>Trained Factor Weights</h3>

    {dataframe_to_html(weight_display)}

    <h2>Out-of-Sample Risk Analysis</h2>

    {dataframe_to_html(risk_display)}

    <h3>Portfolio Turnover</h3>

    {dataframe_to_html(turnover_display)}

    <h2>Latest Factor Ranking</h2>

    <p>
        Latest available ranking date:
        <strong>{latest_ranking_date}</strong>
    </p>

    {dataframe_to_html(ranking_display)}

    <h2>Research Conclusion</h2>

    <div class="warning">

        <p>
            The trained factor specification improved
            upon the original model but did not
            outperform the CSI 300 during the
            out-of-sample period.
        </p>

        <p>
            Its lower Sharpe ratio, larger drawdown
            and high annualised turnover indicate
            that the current factor signal is not
            sufficiently strong for live deployment.
        </p>

        <p>
            The strategy should be treated as a
            quantitative research prototype.
            Future work should expand the stock
            universe, include fundamental factors,
            neutralise sector exposure and apply
            walk-forward validation.
        </p>

    </div>

    <h2>Limitations</h2>

    <ul>
        <li>
            Small stock universe and possible
            survivorship bias.
        </li>

        <li>
            Simplified transaction cost assumptions.
        </li>

        <li>
            No detailed suspension or price-limit model.
        </li>

        <li>
            No sector-neutral portfolio construction.
        </li>

        <li>
            Limited number of out-of-sample months.
        </li>

        <li>
            Results do not guarantee future returns.
        </li>
    </ul>

    <div class="footer">

        Generated automatically by the
        A-Share Multi-Factor Quantitative
        Research System.

        <br>

        For educational and research purposes only.
        This report does not constitute financial advice.

    </div>

</div>

</body>

</html>
"""


    output_path = (
        REPORT_DIR
        / "quant_research_report.html"
    )


    output_path.write_text(
        html_content,
        encoding="utf-8"
    )


    print(
        "Research report saved to:",
        output_path
    )


    return output_path


# =====================================
# 5. 主程序
# =====================================

def main():

    print(
        "\nGenerating final research report..."
    )


    report_path = (
        generate_report()
    )


    print(
        "Opening report in browser..."
    )


    webbrowser.open(
        report_path.resolve().as_uri()
    )


    print(
        "\nReport generation completed."
    )


# =====================================
# 6. 启动程序
# =====================================

if __name__ == "__main__":

    main()