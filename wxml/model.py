"""Train and evaluate: the same model, hyperparameters and metrics as pipeline.py."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from . import config
from .features import FEATURES


def train(train_df: pd.DataFrame, max_iter: int | None = None,
          learning_rate: float | None = None) -> HistGradientBoostingRegressor:
    X, y = train_df[FEATURES], train_df["y"]
    return HistGradientBoostingRegressor(
        max_iter=max_iter or config.MAX_ITER,
        learning_rate=learning_rate or config.LEARNING_RATE,
        random_state=0).fit(X, y)


def evaluate(model, test_df: pd.DataFrame) -> dict:
    """Skill vs persistence (tomorrow = today): 1 - RMSE_model / RMSE_persistence."""
    pred = model.predict(test_df[FEATURES])
    rmse = float(np.sqrt(np.mean((pred - test_df.y) ** 2)))
    rmse_p = float(np.sqrt(np.mean((test_df.tmax_l0 - test_df.y) ** 2)))
    return {"rmse_F": rmse, "rmse_persistence_F": rmse_p,
            "skill": 1 - rmse / rmse_p, "n_test": int(len(test_df))}
