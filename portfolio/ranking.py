from __future__ import annotations

import pandas as pd


def compute_score(p_tp: pd.Series, p_sl: pd.Series) -> pd.Series:
    return (p_tp * (1 - p_sl)).rename("score")


def rank_candidates(df: pd.DataFrame,
                    score_col: str = "score",
                    threshold: float = 0.0) -> pd.DataFrame:
    out = df[df[score_col] > threshold].copy()
    out = out.sort_values(score_col, ascending=False)
    return out