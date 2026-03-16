from __future__ import annotations

import pandas as pd

from src.config import TEST_PATH, TRAIN_PATH, VARIABLE_DEFINITIONS_PATH


def load_train_data() -> pd.DataFrame:
    return pd.read_csv(TRAIN_PATH)


def load_test_data() -> pd.DataFrame:
    return pd.read_csv(TEST_PATH)


def load_variable_definitions() -> pd.DataFrame:
    return pd.read_csv(VARIABLE_DEFINITIONS_PATH)
