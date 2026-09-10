from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "reports" / "results"
REPORT_DIR = PROJECT_ROOT / "reports"
AFTER_COSTS = "Walk-Forward Portfolio After Costs"
BEFORE_COSTS = "Walk-Forward Portfolio Before Costs"
BENCHMARK = "CSI 300 Benchmark"


def load_csv(filename: str) -> pd.DataFrame:
    path = RESULTS_DIR / filename
    assert path.exists(), f"Required file is missing: {path}"
    assert path.stat().st_size > 0, f"File is empty: {path}"
    return pd.read_csv(path, encoding="utf-8-sig")


def find_column(
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
        raise AssertionError(
            f"Missing required column. Expected one of {candidates}. "
            f"Available columns: {list(dataframe.columns)}"
        )

    return None


def numeric_series(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(
        series.astype(str)
        .str.replace("%", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.replace("¥", "", regex=False),
        errors="coerce",
    )

    return values


def weight_series(series: pd.Series) -> pd.Series:
    values = numeric_series(series)

    if not values.dropna().empty:
        if values.dropna().abs().max() > 1.5:
            values = values / 100.0

    return values


def test_required_v2_files_exist():
    required_files = [
        "hs300_constituent_history.csv",
        "stock_master_v2.csv",
        "tradable_universe_v2.csv",
        "raw_factors_v2.csv",
        "processed_factors_v2.csv",
        "factor_ic_summary_v2.csv",
        "walk_forward_factor_weights_v2.csv",
        "walk_forward_stock_scores_v2.csv",
        "portfolio_targets_v2.csv",
        "portfolio_rebalance_orders_v2.csv",
        "execution_price_coverage_v2.csv",
        "event_backtest_daily_v2.csv",
        "event_backtest_trades_v2.csv",
        "event_backtest_performance_v2.csv",
        "event_risk_performance_summary_v2.csv",
    ]

    for filename in required_files:
        path = RESULTS_DIR / filename
        assert path.exists(), f"Missing V2 artifact: {path}"
        assert path.stat().st_size > 0, f"Empty V2 artifact: {path}"


def test_historical_universe_has_point_in_time_coverage():
    universe = load_csv("hs300_constituent_history.csv")

    date_column = find_column(
        universe,
        ["snapshot_date", "date", "query_date"],
    )
    stock_column = find_column(
        universe,
        ["baostock_code", "symbol", "stock_code"],
    )

    universe[date_column] = pd.to_datetime(
        universe[date_column],
        errors="coerce",
    )

    assert universe[date_column].notna().all()
    assert universe[stock_column].notna().all()

    number_of_snapshots = universe[date_column].nunique()
    number_of_stocks = universe[stock_column].nunique()

    assert number_of_snapshots >= 80
    assert number_of_stocks >= 500

    duplicate_count = universe.duplicated(
        subset=[date_column, stock_column]
    ).sum()

    assert duplicate_count == 0


def test_stock_master_contains_historical_and_inactive_stocks():
    stock_master = load_csv("stock_master_v2.csv")

    stock_column = find_column(
        stock_master,
        ["baostock_code", "symbol", "stock_code"],
    )

    assert stock_master[stock_column].nunique() >= 500
    assert not stock_master[stock_column].duplicated().any()

    status_column = find_column(
        stock_master,
        ["listing_status", "is_active", "active"],
        required=False,
    )

    if status_column is not None:
        status = numeric_series(stock_master[status_column])
        inactive_count = (status == 0).sum()
        assert inactive_count >= 1


def test_tradable_universe_has_unique_stock_snapshot_pairs():
    tradability = load_csv("tradable_universe_v2.csv")

    date_column = find_column(
        tradability,
        ["snapshot_date", "date"],
    )
    stock_column = find_column(
        tradability,
        ["baostock_code", "symbol", "stock_code"],
    )

    tradability[date_column] = pd.to_datetime(
        tradability[date_column],
        errors="coerce",
    )

    assert tradability[date_column].nunique() >= 80
    assert tradability[stock_column].nunique() >= 500

    duplicate_count = tradability.duplicated(
        subset=[date_column, stock_column]
    ).sum()

    assert duplicate_count == 0


def test_processed_factor_panel_is_unique():
    factors = load_csv("processed_factors_v2.csv")

    date_column = find_column(
        factors,
        ["snapshot_date", "date"],
    )
    stock_column = find_column(
        factors,
        ["baostock_code", "symbol", "stock_code"],
    )

    duplicate_count = factors.duplicated(
        subset=[date_column, stock_column]
    ).sum()

    assert duplicate_count == 0
    assert factors[date_column].nunique() >= 80
    assert factors[stock_column].nunique() >= 450

    neutralized_columns = [
        column
        for column in factors.columns
        if "neutral" in column.lower()
        and column.lower() != "industry_neutral"
    ]

    assert len(neutralized_columns) >= 5

    for column in neutralized_columns:
        values = pd.to_numeric(
            factors[column],
            errors="coerce",
        )
        assert values.notna().mean() >= 0.95


def test_walk_forward_weights_are_valid():
    weights = load_csv("walk_forward_factor_weights_v2.csv")

    date_column = find_column(
        weights,
        ["snapshot_date", "date", "rebalance_date"],
    )
    component_column = find_column(
        weights,
        ["component_name", "factor_name", "component"],
    )
    weight_column = find_column(
        weights,
        [
            "smoothed_weight",
            "target_weight",
            "trained_weight",
            "weight",
        ],
    )

    weights["_test_weight"] = weight_series(
        weights[weight_column]
    )

    assert weights["_test_weight"].notna().all()
    assert (weights["_test_weight"] >= 0).all()
    assert (weights["_test_weight"] <= 0.36).all()

    duplicate_count = weights.duplicated(
        subset=[date_column, component_column]
    ).sum()

    assert duplicate_count == 0

    weight_sums = weights.groupby(date_column)[
        "_test_weight"
    ].sum()

    active_weight_sums = weight_sums[weight_sums > 1e-8]
    inactive_weight_sums = weight_sums[weight_sums <= 1e-8]

    assert len(active_weight_sums) >= 50

    assert np.allclose(
        active_weight_sums.to_numpy(),
        1.0,
        atol=1e-5,
    )

    assert np.allclose(
        inactive_weight_sums.to_numpy(),
        0.0,
        atol=1e-8,
    )


def test_portfolio_targets_respect_constraints():
    targets = load_csv("portfolio_targets_v2.csv")

    date_column = find_column(
        targets,
        ["snapshot_date", "date", "rebalance_date"],
    )
    stock_column = find_column(
        targets,
        ["baostock_code", "symbol", "stock_code"],
    )
    industry_column = find_column(
        targets,
        ["industry", "industry_name"],
    )
    target_weight_column = find_column(
        targets,
        ["target_weight", "weight"],
    )

    targets["_test_weight"] = weight_series(
        targets[target_weight_column]
    )

    duplicate_count = targets.duplicated(
        subset=[date_column, stock_column]
    ).sum()

    assert duplicate_count == 0
    assert (targets["_test_weight"] > 0).all()

    holdings_per_snapshot = targets.groupby(
        date_column
    )[stock_column].nunique()

    assert (holdings_per_snapshot == 20).all()

    weight_sums = targets.groupby(date_column)[
        "_test_weight"
    ].sum()

    assert np.allclose(
        weight_sums.to_numpy(),
        1.0,
        atol=1e-5,
    )

    maximum_industry_holdings = (
        targets.groupby(
            [date_column, industry_column]
        )[stock_column]
        .nunique()
        .max()
    )

    assert maximum_industry_holdings <= 3


def test_event_backtest_has_no_negative_cash():
    daily = load_csv("event_backtest_daily_v2.csv")

    date_column = find_column(
        daily,
        ["date", "trading_date", "trade_date"],
    )

    dates = pd.to_datetime(
        daily[date_column],
        errors="coerce",
    )

    assert dates.notna().all()
    assert dates.is_monotonic_increasing
    assert not dates.duplicated().any()
    assert len(daily) >= 1_000

    cash_column = find_column(
        daily,
        [
            "cash_after_costs",
            "realistic_cash",
            "portfolio_cash_after_costs",
            "cash",
        ],
        required=False,
    )

    if cash_column is not None:
        cash = pd.to_numeric(
            daily[cash_column],
            errors="coerce",
        )

        assert cash.notna().all()
        assert cash.min() >= -0.01


def test_event_trades_use_board_lots():
    trades = load_csv("event_backtest_trades_v2.csv")

    shares_column = find_column(
        trades,
        ["filled_shares", "executed_shares", "shares"],
        required=False,
    )
    action_column = find_column(
        trades,
        ["action", "side", "trade_action"],
        required=False,
    )

    if shares_column is None or action_column is None:
        return

    actions = trades[action_column].astype(str).str.upper()
    shares = pd.to_numeric(
        trades[shares_column],
        errors="coerce",
    )

    buy_shares = shares[
        (actions == "BUY") & (shares.abs() > 0)
    ].abs().dropna()

    assert not buy_shares.empty

    remainder = np.mod(
        buy_shares.to_numpy(),
        100,
    )

    assert np.allclose(remainder, 0, atol=1e-8)


def test_transaction_costs_reduce_final_value():
    performance = load_csv(
        "event_risk_performance_summary_v2.csv"
    )

    method_column = find_column(performance, ["method"])
    final_value_column = find_column(
        performance,
        ["final_value"],
    )

    values = pd.to_numeric(
        performance.set_index(method_column)[
            final_value_column
        ]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("¥", "", regex=False),
        errors="coerce",
    )

    assert values.notna().all()
    assert BENCHMARK in values.index

    assert values.loc[AFTER_COSTS] > 0
    assert values.loc[BEFORE_COSTS] > 0
    assert values.loc[BENCHMARK] > 0

    assert values.loc[AFTER_COSTS] <= values.loc[BEFORE_COSTS]


def test_drawdown_values_are_valid():
    performance = load_csv(
        "event_risk_performance_summary_v2.csv"
    )

    drawdown_column = find_column(
        performance,
        ["maximum_drawdown_percent"],
    )

    drawdowns = pd.to_numeric(
        performance[drawdown_column],
        errors="coerce",
    )

    assert drawdowns.notna().all()
    assert (drawdowns <= 0).all()
    assert (drawdowns > -100).all()


def test_v2_report_exists_and_is_not_empty():
    html_report = (
        REPORT_DIR
        / "a_share_multi_factor_research_report_v2.html"
    )
    pdf_report = (
        PROJECT_ROOT
        / "A-Share Multi-Factor Research Report V2.pdf"
    )

    assert html_report.exists()
    assert html_report.stat().st_size >= 500_000

    assert pdf_report.exists()
    assert pdf_report.stat().st_size >= 500_000