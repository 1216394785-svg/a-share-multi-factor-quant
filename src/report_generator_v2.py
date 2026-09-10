from __future__ import annotations

import base64
from datetime import datetime
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "reports" / "results"
CHARTS_DIR = PROJECT_ROOT / "reports" / "charts"
REPORT_DIR = PROJECT_ROOT / "reports"

OUTPUT_FILE = (
    REPORT_DIR / "a_share_multi_factor_research_report_v2.html"
)

AFTER_COSTS = "Walk-Forward Portfolio After Costs"
BEFORE_COSTS = "Walk-Forward Portfolio Before Costs"
BENCHMARK = "CSI 300 Benchmark"


def read_csv_safe(filename: str) -> pd.DataFrame:
    path = RESULTS_DIR / filename

    if not path.exists():
        print(f"Warning: missing file {path}")
        return pd.DataFrame()

    return pd.read_csv(path, encoding="utf-8-sig")


def first_existing_column(
    dataframe: pd.DataFrame,
    candidates: list[str],
) -> str | None:
    lookup = {
        str(column).strip().lower(): column
        for column in dataframe.columns
    }

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    return None


def number(value, default: float = np.nan) -> float:
    try:
        converted = float(value)
        return converted if np.isfinite(converted) else default
    except (TypeError, ValueError):
        return default


def format_text(value) -> str:
    if pd.isna(value):
        return "N/A"
    return escape(str(value))


def format_integer(value) -> str:
    numeric = number(value)
    return f"{numeric:,.0f}" if np.isfinite(numeric) else "N/A"


def format_decimal(value, digits: int = 3) -> str:
    numeric = number(value)
    return f"{numeric:.{digits}f}" if np.isfinite(numeric) else "N/A"


def format_percent(value) -> str:
    numeric = number(value)
    return f"{numeric:.2f}%" if np.isfinite(numeric) else "N/A"


def format_weight(value) -> str:
    numeric = number(value)

    if not np.isfinite(numeric):
        return "N/A"

    if abs(numeric) <= 1.5:
        numeric *= 100

    return f"{numeric:.2f}%"


def format_money(value) -> str:
    numeric = number(value)
    return f"¥{numeric:,.2f}" if np.isfinite(numeric) else "N/A"


def format_date(value) -> str:
    converted = pd.to_datetime(value, errors="coerce")

    if pd.isna(converted):
        return "N/A"

    return converted.strftime("%Y-%m-%d")


def make_table(
    dataframe: pd.DataFrame,
    specifications: list[tuple[str, str, callable]],
    maximum_rows: int | None = None,
) -> str:
    if dataframe.empty:
        return '<div class="notice">No data available.</div>'

    available = [
        specification
        for specification in specifications
        if specification[0] in dataframe.columns
    ]

    if not available:
        return '<div class="notice">No matching columns available.</div>'

    display = dataframe.copy()

    if maximum_rows is not None:
        display = display.head(maximum_rows)

    header = "".join(
        f"<th>{escape(label)}</th>"
        for _, label, _ in available
    )

    body_rows = []

    for _, row in display.iterrows():
        cells = []

        for column, _, formatter in available:
            try:
                value = formatter(row[column])
            except Exception:
                value = format_text(row[column])

            cells.append(f"<td>{value}</td>")

        body_rows.append(f"<tr>{''.join(cells)}</tr>")

    return (
        '<div class="table-wrapper">'
        '<table class="data-table">'
        f"<thead><tr>{header}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table>"
        "</div>"
    )


def image_to_data_uri(filename: str) -> str | None:
    path = CHARTS_DIR / filename

    if not path.exists():
        print(f"Warning: missing chart {path}")
        return None

    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def chart_block(
    filename: str,
    title: str,
    description: str,
) -> str:
    data_uri = image_to_data_uri(filename)

    if data_uri is None:
        return ""

    return f"""
    <div class="chart-card">
        <h3>{escape(title)}</h3>
        <p class="chart-description">{escape(description)}</p>
        <img src="{data_uri}" alt="{escape(title)}">
    </div>
    """


def method_row(
    performance: pd.DataFrame,
    method: str,
) -> pd.Series:
    if performance.empty or "method" not in performance.columns:
        return pd.Series(dtype=object)

    matches = performance[performance["method"] == method]

    if matches.empty:
        return pd.Series(dtype=object)

    return matches.iloc[0]


def row_value(
    row: pd.Series,
    column: str,
    default: float = np.nan,
) -> float:
    if row.empty or column not in row.index:
        return default

    return number(row[column], default)


def latest_snapshot(
    dataframe: pd.DataFrame,
    date_candidates: list[str],
) -> tuple[pd.DataFrame, str | None]:
    if dataframe.empty:
        return dataframe, None

    date_column = first_existing_column(
        dataframe,
        date_candidates,
    )

    if date_column is None:
        return dataframe, None

    result = dataframe.copy()
    result[date_column] = pd.to_datetime(
        result[date_column],
        errors="coerce",
    )
    result = result.dropna(subset=[date_column])

    if result.empty:
        return result, date_column

    latest_date = result[date_column].max()
    return result[result[date_column] == latest_date].copy(), date_column


def prepare_factor_ic(
    factor_ic: pd.DataFrame,
) -> pd.DataFrame:
    if factor_ic.empty:
        return factor_ic

    result = factor_ic.copy()

    if "horizon_months" in result.columns:
        horizon = pd.to_numeric(
            result["horizon_months"],
            errors="coerce",
        )
        result = result[horizon == 1]

    if "signal_variant" in result.columns:
        neutralized = result[
            result["signal_variant"]
            .astype(str)
            .str.lower()
            .eq("neutralized")
        ]

        if not neutralized.empty:
            result = neutralized

    if "mean_ic" in result.columns:
        result["mean_ic"] = pd.to_numeric(
            result["mean_ic"],
            errors="coerce",
        )
        result = result.sort_values(
            "mean_ic",
            ascending=False,
        )

    return result.head(10)


def prepare_annual_returns(
    annual_returns: pd.DataFrame,
) -> pd.DataFrame:
    required = {"method", "period", "return_percent"}

    if annual_returns.empty or not required.issubset(
        annual_returns.columns
    ):
        return annual_returns

    pivot = annual_returns.pivot_table(
        index="period",
        columns="method",
        values="return_percent",
        aggfunc="last",
    ).reset_index()

    ordered_columns = [
        "period",
        AFTER_COSTS,
        BEFORE_COSTS,
        BENCHMARK,
    ]

    return pivot[
        [
            column
            for column in ordered_columns
            if column in pivot.columns
        ]
    ]


def build_report() -> Path:
    performance = read_csv_safe(
        "event_risk_performance_summary_v2.csv"
    )
    relative = read_csv_safe(
        "event_relative_performance_v2.csv"
    )
    drawdowns = read_csv_safe(
        "event_drawdown_details_v2.csv"
    )
    annual_returns = read_csv_safe(
        "event_annual_returns_v2.csv"
    )
    turnover_costs = read_csv_safe(
        "event_turnover_cost_summary_v2.csv"
    )

    data_quality = read_csv_safe("data_quality_summary.csv")
    factor_coverage = read_csv_safe("factor_coverage_v2.csv")
    factor_ic = read_csv_safe("factor_ic_summary_v2.csv")
    factor_weights = read_csv_safe(
        "walk_forward_factor_weights_v2.csv"
    )
    portfolio_targets = read_csv_safe(
        "portfolio_targets_v2.csv"
    )
    stock_master = read_csv_safe("stock_master_v2.csv")
    universe_history = read_csv_safe(
        "hs300_constituent_history.csv"
    )

    after_row = method_row(performance, AFTER_COSTS)
    benchmark_row = method_row(performance, BENCHMARK)

    portfolio_return = row_value(
        after_row,
        "total_return_percent",
    )
    benchmark_return = row_value(
        benchmark_row,
        "total_return_percent",
    )
    excess_return = portfolio_return - benchmark_return

    final_value = row_value(after_row, "final_value")
    maximum_drawdown = row_value(
        after_row,
        "maximum_drawdown_percent",
    )
    sharpe_ratio = row_value(after_row, "sharpe_ratio")

    if turnover_costs.empty:
        total_cost_percent = np.nan
    else:
        total_cost_percent = row_value(
            turnover_costs.iloc[0],
            "total_cost_percent_of_initial",
        )

    historical_stocks = (
        len(stock_master)
        if not stock_master.empty
        else np.nan
    )

    snapshot_column = first_existing_column(
        universe_history,
        ["snapshot_date", "date"],
    )

    if snapshot_column is not None:
        historical_snapshots = (
            universe_history[snapshot_column]
            .astype(str)
            .nunique()
        )
    else:
        historical_snapshots = np.nan

    top_factor_ic = prepare_factor_ic(factor_ic)

    latest_weights, weight_date_column = latest_snapshot(
        factor_weights,
        ["snapshot_date", "date", "rebalance_date"],
    )

    latest_portfolio, portfolio_date_column = latest_snapshot(
        portfolio_targets,
        ["snapshot_date", "date", "rebalance_date"],
    )

    if (
        not latest_portfolio.empty
        and "portfolio_position" in latest_portfolio.columns
    ):
        latest_portfolio = latest_portfolio.sort_values(
            "portfolio_position"
        )

    annual_table = prepare_annual_returns(annual_returns)

    performance_table = make_table(
        performance,
        [
            ("method", "Method", format_text),
            ("final_value", "Final Value", format_money),
            (
                "total_return_percent",
                "Total Return",
                format_percent,
            ),
            (
                "annual_return_percent",
                "Annual Return",
                format_percent,
            ),
            (
                "annual_volatility_percent",
                "Annual Volatility",
                format_percent,
            ),
            ("sharpe_ratio", "Sharpe", format_decimal),
            ("sortino_ratio", "Sortino", format_decimal),
            (
                "maximum_drawdown_percent",
                "Maximum Drawdown",
                format_percent,
            ),
            ("calmar_ratio", "Calmar", format_decimal),
            ("var_95_percent", "Daily VaR 95%", format_percent),
            (
                "cvar_95_percent",
                "Daily CVaR 95%",
                format_percent,
            ),
        ],
    )

    relative_table = make_table(
        relative,
        [
            ("comparison", "Comparison", format_text),
            (
                "cumulative_excess_return_percent",
                "Cumulative Excess Return",
                format_percent,
            ),
            (
                "annual_tracking_error_percent",
                "Tracking Error",
                format_percent,
            ),
            (
                "information_ratio",
                "Information Ratio",
                format_decimal,
            ),
            ("beta", "Beta", format_decimal),
            (
                "annual_alpha_percent",
                "Annual Alpha",
                format_percent,
            ),
            (
                "return_correlation",
                "Correlation",
                format_decimal,
            ),
            (
                "up_capture_percent",
                "Up Capture",
                format_percent,
            ),
            (
                "down_capture_percent",
                "Down Capture",
                format_percent,
            ),
        ],
    )

    drawdown_table = make_table(
        drawdowns,
        [
            ("method", "Method", format_text),
            (
                "maximum_drawdown_percent",
                "Maximum Drawdown",
                format_percent,
            ),
            ("peak_date", "Peak", format_date),
            ("trough_date", "Trough", format_date),
            ("recovery_date", "Recovery", format_date),
            (
                "drawdown_trading_days",
                "Peak-to-Trough Days",
                format_integer,
            ),
            (
                "recovery_trading_days",
                "Recovery Days",
                format_integer,
            ),
        ],
    )

    cost_table = make_table(
        turnover_costs,
        [
            (
                "number_of_executed_trades",
                "Executed Trades",
                format_integer,
            ),
            (
                "number_of_rebalances",
                "Rebalances",
                format_integer,
            ),
            (
                "average_monthly_one_way_turnover_percent",
                "Average Monthly Turnover",
                format_percent,
            ),
            (
                "annualized_one_way_turnover_percent",
                "Annualized Turnover",
                format_percent,
            ),
            ("commission", "Commission", format_money),
            ("stamp_duty", "Stamp Duty", format_money),
            ("transfer_fee", "Transfer Fee", format_money),
            ("slippage_cost", "Slippage", format_money),
            (
                "total_transaction_cost",
                "Total Cost",
                format_money,
            ),
            (
                "total_cost_percent_of_initial",
                "Cost / Initial Capital",
                format_percent,
            ),
        ],
    )

    factor_ic_table = make_table(
        top_factor_ic,
        [
            ("factor_name", "Factor", format_text),
            (
                "signal_variant",
                "Variant",
                format_text,
            ),
            (
                "number_of_periods",
                "Periods",
                format_integer,
            ),
            ("mean_ic", "Mean IC", format_decimal),
            ("ic_ir", "IC IR", format_decimal),
            (
                "positive_ic_rate_percent",
                "Positive IC Rate",
                format_percent,
            ),
            (
                "t_statistic",
                "t-Statistic",
                format_decimal,
            ),
        ],
    )

    coverage_table = make_table(
        factor_coverage,
        [
            ("factor_name", "Factor", format_text),
            ("category", "Category", format_text),
            ("role", "Role", format_text),
            (
                "coverage_percent",
                "Coverage",
                format_percent,
            ),
            (
                "first_valid_snapshot",
                "First Valid Snapshot",
                format_date,
            ),
        ],
        maximum_rows=20,
    )

    weights_table = make_table(
        latest_weights,
        [
            ("component_name", "Component", format_text),
            (
                "number_of_training_periods",
                "Training Periods",
                format_integer,
            ),
            (
                "weighted_mean_ic",
                "Weighted Mean IC",
                format_decimal,
            ),
            (
                "positive_ic_rate_percent",
                "Positive IC Rate",
                format_percent,
            ),
            (
                "target_weight",
                "Target Weight",
                format_weight,
            ),
            (
                "smoothed_weight",
                "Smoothed Weight",
                format_weight,
            ),
        ],
    )

    portfolio_table = make_table(
        latest_portfolio,
        [
            (
                "portfolio_position",
                "Position",
                format_integer,
            ),
            ("symbol", "Symbol", format_text),
            ("stock_name", "Stock", format_text),
            ("industry", "Industry", format_text),
            (
                "walk_forward_rank",
                "Model Rank",
                format_integer,
            ),
            (
                "walk_forward_score",
                "Score",
                format_decimal,
            ),
            (
                "selection_status",
                "Status",
                format_text,
            ),
            (
                "target_weight",
                "Weight",
                format_weight,
            ),
        ],
        maximum_rows=30,
    )

    quality_table = make_table(
        data_quality,
        [
            ("check_name", "Quality Check", format_text),
            ("status", "Status", format_text),
            ("issue_count", "Issues", format_integer),
            ("description", "Description", format_text),
        ],
        maximum_rows=20,
    )

    annual_table_html = make_table(
        annual_table,
        [
            ("period", "Year", format_text),
            (
                AFTER_COSTS,
                "Portfolio After Costs",
                format_percent,
            ),
            (
                BEFORE_COSTS,
                "Portfolio Before Costs",
                format_percent,
            ),
            (
                BENCHMARK,
                "CSI 300",
                format_percent,
            ),
        ],
    )

    latest_weight_date = (
        format_date(latest_weights[weight_date_column].iloc[0])
        if not latest_weights.empty
        and weight_date_column is not None
        else "N/A"
    )

    latest_portfolio_date = (
        format_date(
            latest_portfolio[portfolio_date_column].iloc[0]
        )
        if not latest_portfolio.empty
        and portfolio_date_column is not None
        else "N/A"
    )

    main_chart = chart_block(
        "v2_event_backtest_dashboard.png",
        "Portfolio Performance Dashboard",
        "Cumulative value, drawdown, rolling return and "
        "calendar-year performance.",
    )

    risk_chart = chart_block(
        "v2_risk_comparison.png",
        "Performance and Risk Comparison",
        "Comparison of returns, volatility, Sharpe ratio "
        "and maximum drawdown.",
    )

    weight_chart = chart_block(
        "v2_walk_forward_weights.png",
        "Walk-Forward Factor Allocation",
        "Only information available before each rebalance "
        "is used to estimate factor weights.",
    )

    execution_chart = chart_block(
        "v2_execution_costs.png",
        "Turnover and Execution Costs",
        "Transaction-cost decomposition and realized "
        "portfolio turnover.",
    )

    report_date = datetime.now().strftime("%Y-%m-%d")

    css = """
    :root {
        --navy: #102A43;
        --blue: #145DA0;
        --light-blue: #EAF3FA;
        --green: #2E8B57;
        --gold: #D99A2B;
        --red: #C44536;
        --text: #243B53;
        --muted: #627D98;
        --border: #D9E2EC;
        --background: #F4F7FA;
        --white: #FFFFFF;
    }

    * {
        box-sizing: border-box;
    }

    body {
        margin: 0;
        background: var(--background);
        color: var(--text);
        font-family: Arial, "Microsoft YaHei", sans-serif;
        line-height: 1.55;
    }

    .cover {
        min-height: 560px;
        padding: 75px 8%;
        color: white;
        background:
            linear-gradient(
                125deg,
                rgba(16, 42, 67, 0.98),
                rgba(20, 93, 160, 0.92)
            );
        display: flex;
        flex-direction: column;
        justify-content: center;
    }

    .cover-label {
        display: inline-block;
        width: fit-content;
        padding: 7px 14px;
        border: 1px solid rgba(255, 255, 255, 0.45);
        border-radius: 18px;
        font-size: 13px;
        letter-spacing: 1.5px;
        text-transform: uppercase;
    }

    .cover h1 {
        max-width: 950px;
        margin: 24px 0 12px;
        font-size: 46px;
        line-height: 1.12;
    }

    .cover h2 {
        margin: 0;
        max-width: 900px;
        font-size: 22px;
        font-weight: normal;
        color: #D9EAF7;
    }

    .cover-meta {
        margin-top: 55px;
        color: #D9EAF7;
    }

    .container {
        width: min(1220px, 94%);
        margin: 0 auto;
        padding: 36px 0 80px;
    }

    .section {
        margin: 30px 0;
        padding: 30px;
        background: var(--white);
        border: 1px solid var(--border);
        border-radius: 12px;
        box-shadow: 0 4px 18px rgba(16, 42, 67, 0.05);
    }

    .section h2 {
        margin: 0 0 10px;
        color: var(--navy);
        font-size: 25px;
        border-left: 5px solid var(--blue);
        padding-left: 13px;
    }

    .section h3 {
        color: var(--navy);
        margin-top: 24px;
    }

    .section-intro {
        color: var(--muted);
        margin-bottom: 24px;
    }

    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 16px;
        margin-top: 25px;
    }

    .kpi-card {
        padding: 20px;
        border: 1px solid var(--border);
        border-radius: 10px;
        background: linear-gradient(145deg, #FFFFFF, #F5F9FC);
    }

    .kpi-label {
        color: var(--muted);
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .kpi-value {
        margin-top: 7px;
        color: var(--navy);
        font-size: 27px;
        font-weight: bold;
    }

    .positive {
        color: var(--green);
    }

    .negative {
        color: var(--red);
    }

    .pipeline {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 12px;
        margin: 22px 0;
    }

    .pipeline-step {
        position: relative;
        padding: 17px;
        border-radius: 9px;
        color: var(--navy);
        background: var(--light-blue);
        border-top: 4px solid var(--blue);
    }

    .pipeline-number {
        color: var(--blue);
        font-size: 12px;
        font-weight: bold;
    }

    .pipeline-title {
        margin-top: 6px;
        font-weight: bold;
    }

    .pipeline-text {
        margin-top: 5px;
        color: var(--muted);
        font-size: 13px;
    }

    .table-wrapper {
        width: 100%;
        overflow-x: auto;
        margin: 16px 0 25px;
    }

    .data-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 13px;
    }

    .data-table th {
        padding: 11px 10px;
        text-align: left;
        background: var(--navy);
        color: white;
        white-space: nowrap;
    }

    .data-table td {
        padding: 10px;
        border-bottom: 1px solid var(--border);
        vertical-align: top;
    }

    .data-table tbody tr:nth-child(even) {
        background: #F7FAFC;
    }

    .data-table tbody tr:hover {
        background: #EDF5FA;
    }

    .chart-card {
        margin: 24px 0;
        padding: 18px;
        border: 1px solid var(--border);
        border-radius: 10px;
        background: #FFFFFF;
    }

    .chart-card h3 {
        margin-top: 0;
    }

    .chart-description {
        color: var(--muted);
        font-size: 13px;
    }

    .chart-card img {
        display: block;
        width: 100%;
        height: auto;
        margin-top: 12px;
    }

    .callout {
        padding: 18px 20px;
        margin: 20px 0;
        border-left: 5px solid var(--gold);
        background: #FFF9E8;
        border-radius: 5px;
    }

    .notice {
        padding: 14px;
        color: var(--muted);
        background: #F7FAFC;
        border: 1px dashed var(--border);
    }

    .limitations li,
    .method-list li {
        margin-bottom: 9px;
    }

    pre {
        overflow-x: auto;
        padding: 18px;
        color: #E6EDF3;
        background: #0D1B2A;
        border-radius: 8px;
        font-size: 12px;
    }

    .footer {
        padding: 28px 8%;
        text-align: center;
        color: #D9EAF7;
        background: var(--nav);
        font-size: 13px;
    }

    @media (max-width: 850px) {
        .kpi-grid,
        .pipeline {
            grid-template-columns: 1fr;
        }

        .cover h1 {
            font-size: 35px;
        }
    }

    @media print {
        body {
            background: white;
            -webkit-print-color-adjust: exact;
            print-color-adjust: exact;
        }

        .section,
        .chart-card {
            box-shadow: none;
            break-inside: avoid;
        }

        .cover {
            min-height: 95vh;
            break-after: page;
        }

        .container {
            width: 96%;
        }
    }
    """

    html_document = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta
        name="viewport"
        content="width=device-width, initial-scale=1.0"
    >
    <title>A-Share Multi-Factor Research System V2</title>
    <style>{css}</style>
</head>
<body>
    <header class="cover">
        <span class="cover-label">Quantitative Equity Research</span>
        <h1>A-Share Multi-Factor Research System V2</h1>
        <h2>
            Point-in-Time Universe, Neutralized Factors,
            Walk-Forward Modelling and Event-Driven Backtesting
        </h2>
        <div class="cover-meta">
            Research universe: CSI 300 historical constituents<br>
            Report generated: {report_date}<br>
            Framework: Python · SQLite · pandas · matplotlib
        </div>
    </header>

    <main class="container">
        <section class="section">
            <h2>1. Executive Summary</h2>
            <p class="section-intro">
                This report evaluates an end-to-end A-share
                multi-factor research framework designed to reduce
                survivorship bias, look-ahead bias and unrealistic
                execution assumptions.
            </p>

            <div class="kpi-grid">
                <div class="kpi-card">
                    <div class="kpi-label">Historical Stocks</div>
                    <div class="kpi-value">
                        {format_integer(historical_stocks)}
                    </div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Universe Snapshots</div>
                    <div class="kpi-value">
                        {format_integer(historical_snapshots)}
                    </div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Final Portfolio Value</div>
                    <div class="kpi-value">
                        {format_money(final_value)}
                    </div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">After-Cost Return</div>
                    <div class="kpi-value positive">
                        {format_percent(portfolio_return)}
                    </div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Excess vs CSI 300</div>
                    <div class="kpi-value positive">
                        {format_percent(excess_return)}
                    </div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Maximum Drawdown</div>
                    <div class="kpi-value negative">
                        {format_percent(maximum_drawdown)}
                    </div>
                </div>
            </div>

            <div class="callout">
                The portfolio outperformed the CSI 300 over the
                event-backtest period. However, the Sharpe ratio of
                {format_decimal(sharpe_ratio)} indicates that the
                strategy still carries meaningful volatility and
                should not be interpreted as a low-risk investment.
                Explicit costs consumed
                {format_percent(total_cost_percent)} of initial
                capital.
            </div>
        </section>

        <section class="section">
            <h2>2. Research Architecture</h2>
            <p class="section-intro">
                The research process separates data, signals,
                portfolio decisions and execution so that each stage
                can be independently audited.
            </p>

            <div class="pipeline">
                <div class="pipeline-step">
                    <div class="pipeline-number">STEP 01</div>
                    <div class="pipeline-title">Point-in-Time Data</div>
                    <div class="pipeline-text">
                        Historical CSI 300 membership, stock master,
                        market data and tradability filters.
                    </div>
                </div>
                <div class="pipeline-step">
                    <div class="pipeline-number">STEP 02</div>
                    <div class="pipeline-title">Factor Research</div>
                    <div class="pipeline-text">
                        Raw factors, winsorization, standardization,
                        industry and size neutralization.
                    </div>
                </div>
                <div class="pipeline-step">
                    <div class="pipeline-number">STEP 03</div>
                    <div class="pipeline-title">Walk-Forward Model</div>
                    <div class="pipeline-text">
                        Rolling historical IC estimates with no
                        future information used in model weights.
                    </div>
                </div>
                <div class="pipeline-step">
                    <div class="pipeline-number">STEP 04</div>
                    <div class="pipeline-title">Event Execution</div>
                    <div class="pipeline-text">
                        Next-open trading, board lots, liquidity
                        constraints, fees, tax and slippage.
                    </div>
                </div>
            </div>
        </section>

        <section class="section">
            <h2>3. Data and Universe Quality</h2>
            <p class="section-intro">
                The investment universe is reconstructed from
                historical index snapshots. Delisted securities and
                former index constituents remain in the database to
                reduce survivorship bias.
            </p>
            {quality_table}

            <h3>Factor Data Coverage</h3>
            {coverage_table}
        </section>

        <section class="section">
            <h2>4. Factor Evaluation</h2>
            <p class="section-intro">
                Factors are evaluated using cross-sectional Spearman
                information coefficients and forward returns.
                The table shows the strongest one-month neutralized
                signals.
            </p>
            {factor_ic_table}
            {weight_chart}
        </section>

        <section class="section">
            <h2>5. Walk-Forward Factor Model</h2>
            <p class="section-intro">
                Model weights are estimated from historical IC
                observations available before each rebalance.
                Weight smoothing and component limits reduce
                instability and factor concentration.
            </p>

            <h3>Latest Model Weights — {latest_weight_date}</h3>
            {weights_table}

            <ul class="method-list">
                <li>Rolling training window: up to 36 months.</li>
                <li>Minimum training history: 24 months.</li>
                <li>Recent observations receive greater weight.</li>
                <li>Single-component weight is capped.</li>
                <li>Reported look-ahead violations: zero.</li>
            </ul>
        </section>

        <section class="section">
            <h2>6. Portfolio Construction</h2>
            <p class="section-intro">
                The model selects 20 approximately equal-weighted
                stocks. Entry and exit rank buffers reduce unnecessary
                turnover, while industry limits reduce concentration.
            </p>

            <h3>Latest Target Portfolio — {latest_portfolio_date}</h3>
            {portfolio_table}
        </section>

        <section class="section">
            <h2>7. Event-Driven Backtest</h2>
            <p class="section-intro">
                Signals are formed after the snapshot close and
                executed at the next available open. Sells are
                processed before buys, and trades must respect
                available cash and 100-share board lots.
            </p>

            {main_chart}

            <h3>Performance Summary</h3>
            {performance_table}

            <h3>Calendar-Year Returns</h3>
            {annual_table_html}
        </section>

        <section class="section">
            <h2>8. Risk and Relative Performance</h2>
            {risk_chart}

            <h3>Relative Performance</h3>
            {relative_table}

            <h3>Maximum Drawdown Details</h3>
            {drawdown_table}
        </section>

        <section class="section">
            <h2>9. Turnover and Trading Costs</h2>
            <p class="section-intro">
                The realistic simulation includes commission,
                minimum commission, stamp duty, transfer fees and
                bidirectional slippage.
            </p>

            {execution_chart}
            {cost_table}
        </section>

        <section class="section">
            <h2>10. Key Conclusions</h2>
            <ul class="method-list">
                <li>
                    Point-in-time membership and inactive-stock
                    retention materially improve research integrity.
                </li>
                <li>
                    Turnover stability, low-risk factors, value and
                    long-horizon momentum provide the principal
                    model inputs.
                </li>
                <li>
                    The after-cost portfolio achieved
                    {format_percent(portfolio_return)}, compared with
                    {format_percent(benchmark_return)} for the CSI 300.
                </li>
                <li>
                    Transaction costs are economically meaningful and
                    reduce the advantage of frequent rebalancing.
                </li>
                <li>
                    The strategy demonstrates positive relative
                    performance, but its absolute Sharpe ratio and
                    drawdown still require improvement.
                </li>
            </ul>
        </section>

        <section class="section">
            <h2>11. Limitations</h2>
            <ul class="limitations">
                <li>
                    Historical results do not guarantee future
                    performance.
                </li>
                <li>
                    The model uses end-of-day data and does not model
                    intraday order-book liquidity.
                </li>
                <li>
                    Slippage and market impact are simplified and may
                    be higher for large institutional portfolios.
                </li>
                <li>
                    Corporate actions, unavailable data and historical
                    index reconstruction may contain vendor-specific
                    limitations.
                </li>
                <li>
                    Factor selection is based on a limited historical
                    sample and may be exposed to model risk.
                </li>
            </ul>

            <div class="callout">
                This project is provided solely for education and
                quantitative research. It is not investment advice or
                a recommendation to buy or sell securities.
            </div>
        </section>

        <section class="section">
            <h2>12. Reproducibility</h2>
            <p class="section-intro">
                Core V2 research modules can be rerun from the project
                root using the following commands.
            </p>
            <pre>python -m src.data_quality
python -m src.anomaly_review
python -m src.tradability
python -m src.factors_v2
python -m src.factor_processing_v2
python -m src.factor_evaluation_v2
python -m src.walk_forward_model_v2
python -m src.portfolio_construction_v2
python -m src.event_backtester_v2
python -m src.performance_diagnostics_v2
python -m src.visualization_v2
python -m src.report_generator_v2</pre>
        </section>
    </main>

    <footer class="footer">
        A-Share Multi-Factor Research System V2 ·
        Python Quantitative Research Portfolio · {report_date}
    </footer>
</body>
</html>
"""

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(
        html_document,
        encoding="utf-8",
    )

    return OUTPUT_FILE


def main() -> None:
    print("=" * 110)
    print("V2 RESEARCH REPORT GENERATOR")
    print("=" * 110)

    output = build_report()

    print()
    print(f"Report generated: {output}")
    print(
        f"File size: {output.stat().st_size / 1024 / 1024:.2f} MB"
    )
    print()
    print("V2 research report generation completed.")


if __name__ == "__main__":
    main()