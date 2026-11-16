from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class ProbabilityCalibrator:
    def __init__(self, method: str = "isotonic"):
        self.method = method
        self.model = None

    def fit(self, p: pd.Series, y: pd.Series) -> "ProbabilityCalibrator":
        if self.method == "none":
            return self
        if self.method == "isotonic":
            self.model = IsotonicRegression(out_of_bounds="clip")
            self.model.fit(p.values, y.values)
        elif self.method == "sigmoid":
            self.model = LogisticRegression()
            self.model.fit(p.values.reshape(-1, 1), y.values)
        else:
            raise ValueError(f"unknown calibration method: {self.method}")
        return self

    def transform(self, p: pd.Series) -> pd.Series:
        if self.method == "none" or self.model is None:
            return p
        if self.method == "isotonic":
            out = self.model.predict(p.values)
        else:
            out = self.model.predict_proba(p.values.reshape(-1, 1))[:, 1]
        return pd.Series(out, index=p.index, name=p.name)