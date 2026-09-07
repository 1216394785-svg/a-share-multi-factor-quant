# A-Share Multi-Factor Quantitative Research System

A modular quantitative research and backtesting framework for A-share equities, developed with Python, Pandas, SQLite and Matplotlib.

The project covers the complete research process from market data collection and factor construction to portfolio backtesting, out-of-sample validation, risk analysis and automated reporting.

## Project Overview

This project investigates whether price- and volume-based factors can generate stable excess returns within a selected A-share stock universe.

The framework:

- downloads and cleans daily market data;
- stores prices, factors and results in SQLite;
- constructs six quantitative factors;
- performs cross-sectional factor standardisation;
- generates monthly stock rankings;
- selects the top five stocks;
- simulates transaction costs and monthly rebalancing;
- compares the strategy with the CSI 300 Index;
- evaluates individual factors using monthly Rank IC;
- separates training and testing periods;
- performs a true out-of-sample portfolio backtest;
- calculates performance and risk metrics;
- produces CSV reports and visualisations;
- runs automated tests to detect logical errors.

## Technology Stack

- Python
- Pandas
- NumPy
- SQLite
- Matplotlib
- yfinance
- Pytest
- SciPy
- OpenPyXL

## Project Structure

```text
a_share_quant_project
│
├── config
│   ├── __init__.py
│   └── settings.py
│
├── data
│   └── quant_data.db
│
├── reports
│   ├── charts
│   │   └── backtest_report.png
│   └── results
│       ├── annual_returns.csv
│       ├── backtest_results.csv
│       ├── factor_ic_summary.csv
│       ├── factor_validation_summary.csv
│       ├── latest_factor_ranking.csv
│       ├── oos_backtest_results.csv
│       ├── oos_performance_summary.csv
│       ├── oos_risk_summary.csv
│       ├── oos_turnover_summary.csv
│       ├── performance_summary.csv
│       └── pipeline.log
│
├── src
│   ├── __init__.py
│   ├── database.py
│   ├── data_loader.py
│   ├── factors.py
│   ├── factor_analysis.py
│   ├── strategy.py
│   ├── backtester.py
│   ├── performance.py
│   ├── visualization.py
│   ├── ic_analysis.py
│   ├── factor_validation.py
│   ├── oos_backtest.py
│   └── risk_analysis.py
│
├── tests
│   ├── test_settings.py
│   └── test_quant_logic.py
│
├── .gitignore
├── download_data.py
├── main.py
├── requirements.txt
└── README.md
```

## Factor Library

The first version of the model includes six price- and volume-based factors.

| Factor | Definition | Original Direction |
|---|---|---|
| 20-Day Momentum | Price return over the previous 20 trading days | Higher is preferred |
| 60-Day Momentum | Price return over the previous 60 trading days | Higher is preferred |
| 5-Day Reversal | Negative of the previous five-day return | Higher is preferred |
| 20-Day Volatility | Annualised volatility over 20 trading days | Lower is preferred |
| MA Trend | Relative difference between MA20 and MA60 | Higher is preferred |
| Volume Ratio | Current volume relative to its 20-day average | Higher is preferred |

Raw factor values are winsorised and transformed into daily cross-sectional Z-scores.

The original composite score is calculated as:

```text
20-Day Momentum: 20%
60-Day Momentum: 20%
5-Day Reversal: 15%
Low Volatility: 20%
MA Trend: 15%
Volume Ratio: 10%
```

## Portfolio Construction

The portfolio rules are:

- long-only portfolio;
- monthly rebalancing;
- signals calculated at month-end;
- execution delayed until the next trading day;
- top five stocks selected;
- equal weighting;
- maximum individual weight of 20%;
- initial capital of CNY 100,000;
- transaction commissions, stamp duty and slippage included;
- CSI 300 used as the benchmark.

## Avoiding Look-Ahead Bias

The framework applies several controls against future-data leakage:

1. Factor values only use current and historical prices.
2. Month-end signals are executed on the following trading day.
3. Factor direction and weight are estimated only with training data.
4. Trained weights are frozen before out-of-sample testing.
5. Testing results are not used to modify the tested model.
6. Automated tests confirm that training and testing dates do not overlap.

## Factor IC Analysis

Monthly Spearman Rank IC is used to test whether higher factor scores are associated with higher future 20-day returns.

The initial composite factor produced:

```text
Mean monthly Rank IC: -0.0312
Positive IC rate: 43.48%
```

The strongest original factor was 20-day momentum, although its predictive ability remained weak:

```text
Mean IC: 0.0124
Positive IC rate: 52.17%
```

The 60-day momentum and MA trend factors showed negative IC values in the research sample.

## Training and Testing

The available monthly observations were divided chronologically:

```text
Training period: 2020-04-30 to 2024-09-30
Testing period: 2024-10-31 to 2026-06-30
```

Only the training period was used to determine factor direction and factor weight.

The trained composite factor produced:

| Period | Mean IC | Positive IC Rate | t-Statistic |
|---|---:|---:|---:|
| Training | 0.0897 | 54.17% | 1.7970 |
| Testing | 0.0316 | 61.90% | 0.3739 |

The out-of-sample IC remained positive, but the low t-statistic indicates that predictive stability was limited.

## True Out-of-Sample Backtest

The final comparison used 450 out-of-sample trading days.

| Method | Total Return | Annual Return | Annual Volatility | Sharpe Ratio | Maximum Drawdown |
|---|---:|---:|---:|---:|---:|
| Trained Factor | 7.53% | 4.15% | 18.87% | 0.20 | -21.34% |
| Original Factor | -1.08% | -0.61% | 16.60% | -0.07 | -20.95% |
| CSI 300 Benchmark | 16.40% | 8.88% | 16.33% | 0.48 | -13.42% |

The trained factor model improved total return by 8.61 percentage points relative to the original model.

However, it underperformed the CSI 300 by 8.87 percentage points and experienced higher volatility and a larger maximum drawdown.

## Out-of-Sample Risk Results

| Risk Metric | Trained Factor | Original Factor | CSI 300 |
|---|---:|---:|---:|
| 95% Historical VaR | 1.75% | 1.41% | 1.71% |
| 95% Historical CVaR | 2.47% | 2.26% | 2.55% |
| Worst Daily Return | -5.17% | -6.32% | -7.05% |
| Maximum Loss Streak | 10 days | 10 days | 5 days |
| Monthly Win Rate | 52.17% | 43.48% | 60.87% |
| Annualised Turnover | 594.29% | 525.71% | 0.00% |

The trained strategy reduced the worst single-day loss relative to the benchmark but suffered from high turnover and weak risk-adjusted performance.

## Research Conclusion

The trained factor model improved upon the original factor specification, suggesting that factor direction and weighting affected portfolio performance.

However, the model did not outperform the CSI 300 during the true out-of-sample period. Its Sharpe ratio was lower, its maximum drawdown was larger and its annualised turnover was high.

The evidence suggests that the current factors contain limited cross-sectional information, but the signal is not sufficiently stable to generate reliable portfolio-level excess returns.

The strategy should therefore be treated as a research prototype rather than a live trading model.

## Key Limitations

- The stock universe contains only 12 large A-share companies.
- The selected universe may contain survivorship bias.
- Sector exposure is not neutralised.
- Fundamental factors are not included.
- Transaction costs use simplified assumptions.
- Price limits, suspensions and minimum commissions are not fully modelled.
- Portfolio weights are target weights rather than share-level holdings.
- The testing period contains only 21 monthly IC observations.
- Results do not guarantee future performance.

## Installation

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Running the Project

Run the full pipeline with existing market data:

```powershell
python main.py
```

Run without opening charts:

```powershell
python main.py --skip-charts
```

Refresh market data and rerun everything:

```powershell
python main.py --download
```

Run automated tests:

```powershell
python -m pytest -v tests
```

## Automated Testing

The project includes automated checks for:

- duplicate price observations;
- invalid signal and execution dates;
- duplicate rebalance orders;
- portfolio weights above 100%;
- incorrect trained factor weights;
- overlap between training and testing periods;
- missing out-of-sample results;
- negative portfolio values;
- missing daily returns;
- unreasonable single-day returns.

Current test result:

```text
15 passed
```

## Disclaimer

This project is for educational and quantitative research purposes only. It does not constitute financial advice, an investment recommendation or a live trading system.