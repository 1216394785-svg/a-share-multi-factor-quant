"""
A-Share Multi-Factor Quantitative Research System
One-click project pipeline
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable


def run_step(step_number, title, command):
    """Run one project stage and stop if it fails."""

    print("\n" + "=" * 100)
    print(f"STEP {step_number}: {title}")
    print("=" * 100)
    print("Command:", " ".join(command))

    start_time = time.time()

    environment = os.environ.copy()

    # Save charts without blocking the pipeline.
    environment["MPLBACKEND"] = "Agg"

    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=environment,
        check=False,
    )

    elapsed_time = time.time() - start_time

    if result.returncode != 0:
        print("\n" + "!" * 100)
        print(f"STEP FAILED: {title}")
        print(f"Exit code: {result.returncode}")
        print("The pipeline has stopped.")
        print("!" * 100)
        raise SystemExit(result.returncode)

    print(f"\nCompleted in {elapsed_time:.2f} seconds.")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run the A-share multi-factor quant research project."
    )

    parser.add_argument(
        "--download",
        action="store_true",
        help="Download and update market data before running the analysis.",
    )

    parser.add_argument(
        "--skip-charts",
        action="store_true",
        help="Skip full-sample and out-of-sample chart generation.",
    )

    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip automated tests.",
    )

    parser.add_argument(
        "--skip-report",
        action="store_true",
        help="Skip HTML research report generation.",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    print("\n" + "=" * 100)
    print("A-SHARE MULTI-FACTOR QUANTITATIVE RESEARCH SYSTEM")
    print("=" * 100)
    print(f"Project directory: {PROJECT_ROOT}")
    print(f"Python interpreter: {PYTHON}")
    print(f"Update market data: {args.download}")
    print(f"Generate charts: {not args.skip_charts}")
    print(f"Run tests: {not args.skip_tests}")
    print(f"Generate report: {not args.skip_report}")

    steps = []

    steps.append(
        (
            "Initialize database",
            [PYTHON, "-m", "src.database"],
        )
    )

    if args.download:
        steps.append(
            (
                "Download market data",
                [PYTHON, "download_data.py"],
            )
        )

    steps.extend(
        [
            (
                "Calculate raw factor values",
                [PYTHON, "-m", "src.factors"],
            ),
            (
                "Standardize and rank factors",
                [PYTHON, "-m", "src.factor_analysis"],
            ),
            (
                "Generate portfolio rebalance orders",
                [PYTHON, "-m", "src.strategy"],
            ),
            (
                "Run full-sample portfolio backtest",
                [PYTHON, "-m", "src.backtester"],
            ),
            (
                "Analyze full-sample performance",
                [PYTHON, "-m", "src.performance"],
            ),
            (
                "Calculate factor information coefficients",
                [PYTHON, "-m", "src.ic_analysis"],
            ),
            (
                "Train and validate factor weights",
                [PYTHON, "-m", "src.factor_validation"],
            ),
            (
                "Run out-of-sample backtest",
                [PYTHON, "-m", "src.oos_backtest"],
            ),
            (
                "Analyze out-of-sample risk",
                [PYTHON, "-m", "src.risk_analysis"],
            ),
        ]
    )

    if not args.skip_charts:
        steps.extend(
            [
                (
                    "Generate full-sample charts",
                    [PYTHON, "-m", "src.visualization"],
                ),
                (
                    "Generate out-of-sample charts",
                    [PYTHON, "-m", "src.oos_visualization"],
                ),
            ]
        )

    if not args.skip_tests:
        steps.append(
            (
                "Run automated tests",
                [PYTHON, "-m", "pytest", "-v", "tests"],
            )
        )

    if not args.skip_report:
        steps.append(
            (
                "Generate quantitative research report",
                [PYTHON, "-m", "src.report_generator"],
            )
        )

    pipeline_start = time.time()

    for step_number, (title, command) in enumerate(steps, start=1):
        run_step(step_number, title, command)

    total_time = time.time() - pipeline_start

    print("\n" + "=" * 100)
    print("PROJECT PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 100)
    print(f"Total running time: {total_time:.2f} seconds")
    print(f"Research report: {PROJECT_ROOT / 'reports' / 'quant_research_report.html'}")
    print(f"Chart directory: {PROJECT_ROOT / 'reports' / 'charts'}")
    print(f"Result directory: {PROJECT_ROOT / 'reports' / 'results'}")
    print("=" * 100)


if __name__ == "__main__":
    main()