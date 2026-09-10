from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "reports" / "results"
CHARTS_DIR = PROJECT_ROOT / "reports" / "charts"

DAILY_FILE = RESULTS_DIR / "event_backtest_daily_v2.csv"
PERFORMANCE_FILE = (
    RESULTS_DIR / "event_risk_performance_summary_v2.csv"
)
ROLLING_FILE = RESULTS_DIR / "event_rolling_metrics_v2.csv"
ANNUAL_FILE = RESULTS_DIR / "event_annual_returns_v2.csv"
COST_FILE = RESULTS_DIR / "event_turnover_cost_summary_v2.csv"
WEIGHTS_FILE = RESULTS_DIR / "walk_forward_factor_weights_v2.csv"

INITIAL_CAPITAL = 1_000_000.0

AFTER_COSTS = "Walk-Forward Portfolio After Costs"
BEFORE_COSTS = "Walk-Forward Portfolio Before Costs"
BENCHMARK = "CSI 300 Benchmark"

METHOD_COLORS = {
    AFTER_COSTS: "#145DA0",
    BEFORE_COSTS: "#2E8B57",
    BENCHMARK: "#808080",
}

METHOD_COLUMNS = {
    AFTER_COSTS: [
        "portfolio_value_after_costs",
        "after_cost_portfolio_value",
        "portfolio_after_costs",
        "realistic_portfolio_value",
        "realistic_total_value",
        "realistic_value",
        "portfolio_value",
    ],
    BEFORE_COSTS: [
        "portfolio_value_before_costs",
        "before_cost_portfolio_value",
        "portfolio_before_costs",
        "zero_cost_portfolio_value",
        "zero_cost_total_value",
        "zero_cost_value",
    ],
    BENCHMARK: [
        "benchmark_value",
        "csi300_value",
        "benchmark_portfolio_value",
        "benchmark_total_value",
    ],
}


def set_chart_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "#F7F9FC",
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#C9D2DD",
            "axes.labelcolor": "#25364A",
            "axes.titlecolor": "#172B4D",
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "font.size": 10,
            "font.family": "DejaVu Sans",
            "xtick.color": "#425466",
            "ytick.color": "#425466",
            "grid.color": "#DCE3EB",
            "grid.alpha": 0.65,
            "legend.frameon": False,
            "lines.linewidth": 2.0,
        }
    )


def resolve_column(
    dataframe: pd.DataFrame,
    candidates: list[str],
    required: bool = True,
) -> str | None:
    lookup = {
        str(column).strip().lower(): column
        for column in dataframe.columns
    }

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    if required:
        raise KeyError(
            f"Could not find columns {candidates}.\n"
            f"Available columns: {list(dataframe.columns)}"
        )

    return None


def last_numeric_value(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()

    if values.empty:
        return np.nan

    return float(values.iloc[-1])


def load_nav_data(
    performance: pd.DataFrame,
) -> dict[str, pd.Series]:
    if not DAILY_FILE.exists():
        raise FileNotFoundError(f"Missing file: {DAILY_FILE}")

    daily = pd.read_csv(DAILY_FILE)

    date_column = resolve_column(
        daily,
        ["date", "trading_date", "trade_date"],
    )

    daily[date_column] = pd.to_datetime(
        daily[date_column],
        errors="coerce",
    )
    daily = daily.dropna(subset=[date_column])
    daily = daily.sort_values(date_column)
    daily = daily.drop_duplicates(date_column, keep="last")

    method_column = resolve_column(
        daily,
        ["method", "portfolio_method"],
        required=False,
    )
    value_column = resolve_column(
        daily,
        ["value", "portfolio_value", "nav"],
        required=False,
    )

    if method_column is not None and value_column is not None:
        pivot = daily.pivot_table(
            index=date_column,
            columns=method_column,
            values=value_column,
            aggfunc="last",
        )

        if all(method in pivot.columns for method in METHOD_COLORS):
            return {
                method: pd.to_numeric(
                    pivot[method],
                    errors="coerce",
                ).dropna()
                for method in METHOD_COLORS
            }

    resolved: dict[str, str] = {}
    used_columns: set[str] = set()

    column_lookup = {
        str(column).strip().lower(): column
        for column in daily.columns
    }

    for method, candidates in METHOD_COLUMNS.items():
        for candidate in candidates:
            matched = column_lookup.get(candidate.lower())

            if matched is not None and matched not in used_columns:
                resolved[method] = matched
                used_columns.add(matched)
                break

    final_values = performance.set_index("method")["final_value"]

    numeric_candidates = []

    for column in daily.columns:
        if column == date_column or column in used_columns:
            continue

        numeric_values = pd.to_numeric(
            daily[column],
            errors="coerce",
        )

        if numeric_values.notna().sum() >= len(daily) * 0.8:
            final_value = last_numeric_value(numeric_values)

            if pd.notna(final_value) and final_value > 10_000:
                numeric_candidates.append(column)

    for method in METHOD_COLORS:
        if method in resolved:
            continue

        target_value = float(final_values.loc[method])
        best_column = None
        best_error = np.inf

        for column in numeric_candidates:
            if column in used_columns:
                continue

            candidate_final = last_numeric_value(daily[column])
            relative_error = abs(
                candidate_final - target_value
            ) / max(abs(target_value), 1.0)

            if relative_error < best_error:
                best_error = relative_error
                best_column = column

        if best_column is None or best_error > 0.10:
            raise KeyError(
                f"Unable to identify the NAV column for {method}.\n"
                f"Available columns: {list(daily.columns)}"
            )

        resolved[method] = best_column
        used_columns.add(best_column)

    nav_data: dict[str, pd.Series] = {}

    for method, column in resolved.items():
        nav = pd.Series(
            pd.to_numeric(
                daily[column],
                errors="coerce",
            ).to_numpy(),
            index=pd.DatetimeIndex(daily[date_column]),
            name=method,
        )

        nav = nav.dropna()
        nav = nav[nav > 0]
        nav = nav[~nav.index.duplicated(keep="last")]
        nav_data[method] = nav.sort_index()

    return nav_data


def style_axis(axis: plt.Axes) -> None:
    axis.grid(True, axis="y")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def add_panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.08,
        1.06,
        label,
        transform=axis.transAxes,
        fontsize=13,
        fontweight="bold",
               color="#172B4D",
    )


def save_figure(
    figure: plt.Figure,
    output_path: Path,
) -> None:
    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        facecolor=figure.get_facecolor(),
    )
    plt.close(figure)


def calculate_drawdown(nav: pd.Series) -> pd.Series:
    running_maximum = np.maximum.accumulate(
        np.concatenate([[INITIAL_CAPITAL], nav.to_numpy()])
    )[1:]

    return pd.Series(
        nav.to_numpy() / running_maximum - 1.0,
        index=nav.index,
    )


def create_main_dashboard(
    nav_data: dict[str, pd.Series],
    rolling: pd.DataFrame,
    annual: pd.DataFrame,
) -> Path:
    figure, axes = plt.subplots(
        2,
               2,
               figsize=(16, 10),
    )
    figure.suptitle(
        "V2 Walk-Forward Event Portfolio — Event-Driven Backtest",
        fontsize=19,
        fontweight="bold",
        color="#172B4D",
        y=0.99,
    )

    nav_axis = axes[0, 0]

    for method, nav in nav_data.items():
        normalized = nav / INITIAL_CAPITAL * 100

        nav_axis.plot(
            normalized.index,
            normalized,
            label=method,
            color=METHOD_COLORS[method],
        )

    nav_axis.axhline(
        100,
        color="#9AA5B1",
        linestyle="--",
        linewidth=1,
    )
    nav_axis.set_title("Cumulative Portfolio Value")
    nav_axis.set_ylabel("Initial capital = 100")
    nav_axis.legend(fontsize=8)
    style_axis(nav_axis)
    add_panel_label(nav_axis, "A")

    drawdown_axis = axes[0, 1]

    for method, nav in nav_data.items():
        drawdown = calculate_drawdown(nav) * 100

        drawdown_axis.plot(
            drawdown.index,
            drawdown,
            label=method,
            color=METHOD_COLORS[method],
        )

        if method == AFTER_COSTS:
            drawdown_axis.fill_between(
                drawdown.index,
                drawdown.to_numpy(),
                0,
                color=METHOD_COLORS[method],
                alpha=0.12,
            )

    drawdown_axis.axhline(0, color="#9AA5B1", linewidth=1)
    drawdown_axis.set_title("Portfolio Drawdown")
    drawdown_axis.set_ylabel("Drawdown (%)")
    style_axis(drawdown_axis)
    add_panel_label(drawdown_axis, "B")

    rolling_axis = axes[1, 0]
    rolling["date"] = pd.to_datetime(
        rolling["date"],
        errors="coerce",
    )

    for method in METHOD_COLORS:
        subset = rolling[rolling["method"] == method].dropna(
            subset=["date", "rolling_12m_return_percent"]
        )

        rolling_axis.plot(
            subset["date"],
            subset["rolling_12m_return_percent"],
            label=method,
            color=METHOD_COLORS[method],
        )

    rolling_axis.axhline(
        0,
        color="#9AA5B1",
        linestyle="--",
        linewidth=1,
    )
    rolling_axis.set_title("Rolling 12-Month Return")
    rolling_axis.set_ylabel("Return (%)")
    style_axis(rolling_axis)
    add_panel_label(rolling_axis, "C")

    annual_axis = axes[1, 1]
    annual_pivot = annual.pivot(
        index="period",
        columns="method",
        values="return_percent",
    )

    available_methods = [
        method
        for method in METHOD_COLORS
        if method in annual_pivot.columns
    ]

    annual_pivot[available_methods].plot(
        kind="bar",
        ax=annual_axis,
        color=[
            METHOD_COLORS[method]
            for method in available_methods
        ],
        width=0.78,
    )

    annual_axis.axhline(0, color="#9AA5B1", linewidth=1)
    annual_axis.set_title("Calendar-Year Returns")
    annual_axis.set_xlabel("")
    annual_axis.set_ylabel("Return (%)")
    annual_axis.tick_params(axis="x", rotation=0)
    annual_axis.legend(fontsize=7)
    style_axis(annual_axis)
    add_panel_label(annual_axis, "D")

    for axis in axes.flat:
        axis.xaxis.set_major_locator(mdates.AutoDateLocator())
        if not isinstance(axis.get_xticks()[0], str):
            pass

    figure.text(
        0.01,
        0.01,
        "Note: 2022 and 2026 are partial years. "
        "Signals are executed at the next available market open.",
        fontsize=8,
        color="#5D6D7E",
    )

    figure.tight_layout(rect=[0, 0.035, 1, 0.96])

    output = CHARTS_DIR / "v2_event_backtest_dashboard.png"
    save_figure(figure, output)
    return output


def add_bar_labels(
    axis: plt.Axes,
    bars,
    suffix: str = "",
) -> None:
    for bar in bars:
        height = bar.get_height()

        if not np.isfinite(height):
            continue

        vertical_alignment = "bottom" if height >= 0 else "top"
        offset = max(abs(height) * 0.025, 0.08)

        label_y = (
            height + offset
            if height >= 0
            else height - offset
        )

        axis.text(
            bar.get_x() + bar.get_width() / 2,
            label_y,
            f"{height:.2f}{suffix}",
            ha="center",
            va=vertical_alignment,
            fontsize=8,
            color="#25364A",
        )


def create_risk_comparison(
    performance: pd.DataFrame,
) -> Path:
    metrics = [
        ("total_return_percent", "Total Return", "%", False),
        (
            "annual_volatility_percent",
            "Annual Volatility",
            "%",
            False,
        ),
        ("sharpe_ratio", "Sharpe Ratio", "", False),
        (
            "maximum_drawdown_percent",
            "Maximum Drawdown",
            "%",
            True,
        ),
    ]

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(14, 9),
    )

    figure.suptitle(
        "V2 Performance and Risk Comparison",
        fontsize=18,
        fontweight="bold",
        color="#172B4D",
    )

    methods = [
        method
        for method in METHOD_COLORS
        if method in performance["method"].values
    ]

    short_names = {
        AFTER_COSTS: "After Costs",
        BEFORE_COSTS: "Before Costs",
        BENCHMARK: "CSI 300",
    }

    for axis, metric_info, panel in zip(
        axes.flat,
        metrics,
        ["A", "B", "C", "D"],
    ):
        metric, title, suffix, absolute_value = metric_info

        values = (
            performance.set_index("method")
            .loc[methods, metric]
            .astype(float)
        )

        if absolute_value:
            values = values.abs()

        bars = axis.bar(
            [short_names[method] for method in methods],
            values,
            color=[METHOD_COLORS[method] for method in methods],
            width=0.62,
        )

        axis.axhline(0, color="#9AA5B1", linewidth=1)
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=8)
        style_axis(axis)
        add_panel_label(axis, panel)
        add_bar_labels(axis, bars, suffix)

    figure.tight_layout(rect=[0, 0, 1, 0.95])

    output = CHARTS_DIR / "v2_risk_comparison.png"
    save_figure(figure, output)
    return output


def convert_weight_column(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(
        series.astype(str)
        .str.replace("%", "", regex=False)
        .str.replace(",", "", regex=False),
        errors="coerce",
    )

    if values.dropna().empty:
        return values

    if values.dropna().abs().max() > 1.5:
        values = values / 100.0

    return values


def create_weight_chart() -> Path | None:
    if not WEIGHTS_FILE.exists():
        print(f"Weight chart skipped: {WEIGHTS_FILE} not found.")
        return None

    weights = pd.read_csv(WEIGHTS_FILE)

    date_column = resolve_column(
        weights,
        ["snapshot_date", "date", "rebalance_date"],
    )
    component_column = resolve_column(
        weights,
        ["component_name", "factor_name", "component"],
    )
    weight_column = resolve_column(
        weights,
        [
            "smoothed_weight",
            "trained_weight",
            "target_weight",
            "weight",
        ],
    )

    weights[date_column] = pd.to_datetime(
        weights[date_column],
        errors="coerce",
    )
    weights["_weight"] = convert_weight_column(
        weights[weight_column]
    )

    weights = weights.dropna(
        subset=[date_column, component_column, "_weight"]
    )

    pivot = weights.pivot_table(
        index=date_column,
        columns=component_column,
        values="_weight",
        aggfunc="last",
    ).sort_index()

    pivot = pivot.fillna(0.0)
    row_totals = pivot.sum(axis=1).replace(0, np.nan)
    pivot = pivot.div(row_totals, axis=0).fillna(0.0)

    colors = [
        "#145DA0",
        "#2E8B57",
        "#E39A2D",
        "#7A5195",
        "#D45087",
        "#4C78A8",
        "#72B7B2",
    ]

    figure, axis = plt.subplots(figsize=(15, 7))

    axis.stackplot(
        pivot.index,
        *[pivot[column] * 100 for column in pivot.columns],
        labels=list(pivot.columns),
        colors=colors[: len(pivot.columns)],
        alpha=0.9,
    )

    axis.set_title(
        "Walk-Forward Factor Weights Through Time",
        fontsize=17,
        pad=15,
    )
    axis.set_ylabel("Portfolio factor weight (%)")
    axis.set_ylim(0, 100)
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.10),
        ncol=3,
        fontsize=9,
    )
    style_axis(axis)
    figure.tight_layout()

    output = CHARTS_DIR / "v2_walk_forward_weights.png"
    save_figure(figure, output)
    return output


def create_execution_chart() -> Path | None:
    if not COST_FILE.exists():
        print(f"Execution chart skipped: {COST_FILE} not found.")
        return None

    costs = pd.read_csv(COST_FILE)

    if costs.empty:
        return None

    row = costs.iloc[0]

    cost_components = {
        "Commission": float(row.get("commission", 0.0)),
        "Stamp Duty": float(row.get("stamp_duty", 0.0)),
        "Transfer Fee": float(row.get("transfer_fee", 0.0)),
        "Slippage": float(row.get("slippage_cost", 0.0)),
    }

    turnover_values = {
        "Average Monthly": float(
            row.get(
                "average_monthly_one_way_turnover_percent",
                0.0,
            )
        ),
        "Annualized": float(
            row.get(
                "annualized_one_way_turnover_percent",
                0.0,
            )
        ),
    }

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(14, 6),
    )

    cost_bars = axes[0].bar(
        list(cost_components.keys()),
        list(cost_components.values()),
        color=["#145DA0", "#D45087", "#72B7B2", "#E39A2D"],
        width=0.62,
    )

    axes[0].set_title("Transaction-Cost Decomposition")
    axes[0].set_ylabel("Cost (CNY)")
    axes[0].tick_params(axis="x", rotation=12)
    style_axis(axes[0])

    for bar in cost_bars:
        value = bar.get_height()
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:,.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    turnover_bars = axes[1].bar(
        list(turnover_values.keys()),
        list(turnover_values.values()),
        color=["#2E8B57", "#7A5195"],
        width=0.55,
    )

    axes[1].set_title("Realized One-Way Turnover")
    axes[1].set_ylabel("Turnover (%)")
    style_axis(axes[1])
    add_bar_labels(axes[1], turnover_bars, "%")

    total_cost = float(
        row.get("total_transaction_cost", 0.0)
    )
    number_of_trades = int(
        float(row.get("number_of_executed_trades", 0))
    )

    figure.suptitle(
        f"Execution Analysis — {number_of_trades:,} Trades, "
        f"CNY {total_cost:,.0f} Total Estimated Cost",
        fontsize=16,
        fontweight="bold",
        color="#172B4D",
    )

    figure.tight_layout(rect=[0, 0, 1, 0.93])

    output = CHARTS_DIR / "v2_execution_costs.png"
    save_figure(figure, output)
    return output


def main() -> None:
    print("=" * 110)
    print("V2 PROFESSIONAL VISUALIZATION")
    print("=" * 110)

    set_chart_style()
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    performance = pd.read_csv(PERFORMANCE_FILE)
    rolling = pd.read_csv(ROLLING_FILE)
    annual = pd.read_csv(ANNUAL_FILE)

    nav_data = load_nav_data(performance)

    outputs = [
        create_main_dashboard(
            nav_data=nav_data,
            rolling=rolling,
            annual=annual,
        ),
        create_risk_comparison(performance),
        create_weight_chart(),
        create_execution_chart(),
    ]

    print()
    print("=" * 110)
    print("OUTPUT CHARTS")
    print("=" * 110)

    for output in outputs:
        if output is not None:
            print(output)

    print()
    print("V2 visualization completed.")


if __name__ == "__main__":
    main()