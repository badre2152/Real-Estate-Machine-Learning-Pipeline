"""Prepare DVC training and evaluation feature datasets."""

import pickle
from pathlib import Path

import pandas as pd

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
    train, geo_stats = engineer_features_train(train)
    test = engineer_features_test(test, geo_stats)
    if "categorie_prix" in train.columns and "categorie_prix" not in test.columns:
        test = add_classification_target(test)

    train.to_parquet(output_dir / "train_fe.parquet", index=False)
    test.to_parquet(output_dir / "test_fe.parquet", index=False)
    with (output_dir / "geo_stats.pkl").open("wb") as output:
        pickle.dump(geo_stats, output)
    logger.info("DVC features saved: %s train rows, %s test rows", len(train), len(test))


if __name__ == "__main__":
    main()
