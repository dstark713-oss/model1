from __future__ import annotations

import pandas as pd
from catboost import CatBoostClassifier


class SLModel:
    def __init__(self, params: dict):
        self.params = dict(params)
        self.model: CatBoostClassifier | None = None
        self.feature_names: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SLModel":
        self.feature_names = list(X.columns)
        self.model = CatBoostClassifier(**self.params)
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        p = self.model.predict_proba(X[self.feature_names])[:, 1]
        return pd.Series(p, index=X.index, name="p_sl")