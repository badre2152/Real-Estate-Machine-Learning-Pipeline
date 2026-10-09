"""Prepare DVC training and evaluation feature datasets."""

import pickle
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from config_loader import cfg
from features import add_classification_target, engineer_features_test, engineer_features_train
from logger_setup import get_logger
from prepare import clean_dataframe, split_data

logger = get_logger(__name__)


def main() -> None:
    source = Path("data/raw/obt.parquet")
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    dataframe = clean_dataframe(pd.read_parquet(source))
    train, test = split_data(
        dataframe,
        test_size=float(cfg.pipeline.test_size),
        random_state=int(cfg.pipeline.random_state),
    )
    calibration = None
    if cfg.prediction_intervals.method == "quantile":
        if len(train) < 12:
            raise ValueError("Not enough rows for calibration")
        train, calibration = train_test_split(
            train, test_size=0.2, random_state=int(cfg.pipeline.random_state)
        )
        train = train.reset_index(drop=True)
        calibration = calibration.reset_index(drop=True)
    train, geo_stats = engineer_features_train(train)
    test = engineer_features_test(test, geo_stats)
    if calibration is not None:
        calibration = engineer_features_test(calibration, geo_stats)
    if "categorie_prix" in train.columns and "categorie_prix" not in test.columns:
        test = add_classification_target(test)

    train.to_parquet(output_dir / "train_fe.parquet", index=False)
    test.to_parquet(output_dir / "test_fe.parquet", index=False)
    if calibration is not None:
        calibration.to_parquet(output_dir / "calibration_fe.parquet", index=False)
    else:
        (output_dir / "calibration_fe.parquet").unlink(missing_ok=True)
    with (output_dir / "geo_stats.pkl").open("wb") as output:
        pickle.dump(geo_stats, output)
    logger.info("DVC features saved: %s train rows, %s test rows", len(train), len(test))


if __name__ == "__main__":
    main()
