from __future__ import annotations

import numpy as np
import pandas as pd


def select_n_on_train(train_df: pd.DataFrame,
                      cols: list[str],
                      mcfg: dict,
                      pcfg: dict) -> int:
    """
    На train: последние 20% по времени — внутренняя validation.
    Обучаем модели на 80%, предсказываем на 20%.
    Для каждого N из grid: средний score top-N по дням.
    N* — максимум.
    """
    from models.tp_model import TPModel
    from models.sl_model import SLModel
    from portfolio.ranking import compute_score

    train_df = train_df.sort_values("date")
    cut = int(len(train_df) * 0.8)
    fit_df, val_df = train_df.iloc[:cut], train_df.iloc[cut:]
    if val_df.empty:
        return 4

    Xfit = fit_df[cols]
    Xval = val_df[cols]
    tp = TPModel(mcfg["models"]["tp"]).fit(Xfit, fit_df["tp_hit"].astype(int))
    sl = SLModel(mcfg["models"]["sl"]).fit(Xfit, fit_df["sl_hit"].astype(int))

    val = val_df.copy()
    val["p_tp"] = tp.predict_proba(Xval).values
    val["p_sl"] = sl.predict_proba(Xval).values
    val["score"] = compute_score(val["p_tp"], val["p_sl"])

    grid = list(pcfg["selection"]["n_assets_grid"])
    scores_by_n = {n: [] for n in grid}

    for _, day in val.groupby("date"):
        day = day.set_index("ticker")
        for n in grid:
            cand = day.sort_values("score", ascending=False).head(n)
            if cand.empty:
                continue
            scores_by_n[n].append(float(cand["score"].mean()))

    best_n, best = 4, -np.inf
    for n, arr in scores_by_n.items():
        if not arr:
            continue
        m = float(np.mean(arr))
        if m > best:
            best, best_n = m, n
    return best_n