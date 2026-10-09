"""Run the DVC data validation stage and save its JSON report."""

import argparse
from pathlib import Path

import pandas as pd

from data_validation import DataValidator
from logger_setup import get_logger

logger = get_logger(__name__)

def main() -> int:
    parser = argparse.ArgumentParser(description="Validate an extracted Parquet dataset")
    parser.add_argument("--input", type=Path, default=Path("data/raw/obt.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()

    dataframe = pd.read_parquet(args.input)
    validator = DataValidator()
    report = validator.validate(dataframe, stage="dvc_validate")
    validator.save_report(report, output_dir=str(args.output_dir))

    if not report.passed:
        logger.error("DVC data validation failed")
        return 1

    logger.info("DVC data validation completed")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
