from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import yaml

from data.storage import SqliteStorage, StorageConfig
from features.pipeline import compute_features
from models.persistence import load_latest
from portfolio.allocation import equal_weights
from portfolio.breadth import breadth_ratio
from portfolio.correlation import precompute_returns, window_corr, greedy_filter
from portfolio.ranking import compute_score
from portfolio.regime import compute_regime

log = logging.getLogger(__name__)


def _resolve(p: str) -> Path:
    p = Path(p)
    return p if p.is_absolute() else Path(__file__).resolve().parent.parent / p


def _build_today_features(st: SqliteStorage, tickers: list[str],
                          fcfg: dict, last_date: pd.Timestamp) -> pd.DataFrame:
    rows = []
    for t in tickers:
        px = st.load(t)
        if px.empty or len(px) < 300:
            continue
        feats = compute_features(px, fcfg)
        if feats.empty:
            continue
        sub = feats.loc[feats.index <= last_date]
        if sub.empty:
            continue
        row = sub.loc[[sub.index.max()]].copy()
        row["ticker"] = t
        row.index.name = "date"
        rows.append(row.reset_index())
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def build_signal(st: SqliteStorage, cfg: dict, fcfg: dict,
                 mcfg: dict, pcfg: dict) -> dict:
    loaded = load_latest(_resolve(mcfg["persistence"]["models_dir"]))
    if loaded is None:
        raise SystemExit("no model version found, run retrain.py first")
    mv, tp_model, sl_model, cal_tp, cal_sl = loaded
    log.info("[signal] model version=%s retrain_date=%s",
             mv.version, mv.retrain_date)

    imoex = st.load("IMOEX")
    regime = compute_regime(imoex, ema_period=cfg["regime"]["ema_period"]).dropna()

    top = pd.read_parquet(_resolve("artifacts/phase2/top100_pit.parquet"))
    top_last = top["date"].max()
    tickers = top.loc[top["date"] == top_last, "ticker"].tolist()

    # общая последняя торговая дата: есть и режим, и TOP100
    last_date = min(regime.index.max(), top_last)
    active = str(regime.loc[last_date])
    log.info("[signal] last_date=%s regime=%s", last_date.date(), active)
    log.info("[signal] universe=stocks tickers=%d (%s)",
             len(tickers), top_last.date())

    
    close_by_ticker = {}
    for t in tickers:
        df = st.load(t)
        if df.empty:
            continue
        close_by_ticker[t] = df.set_index("date")["close"]
    b = breadth_ratio(close_by_ticker, pcfg["allocation"]["breadth_ema"]).dropna()
    exposure = float(b.get(last_date, 0.0)) if not b.empty else 0.0
    log.info("[signal] breadth=%.3f", exposure)

    today_df = _build_today_features(st, tickers, fcfg, last_date)
    if today_df.empty:
        raise SystemExit("no features on last date")

    cols = mv.feature_names
    today_df = today_df.dropna(subset=cols, how="any")
    Xte = today_df[cols]

    p_tp = pd.Series(tp_model.predict_proba(Xte)[:, 1], index=today_df.index)
    p_sl = pd.Series(sl_model.predict_proba(Xte)[:, 1], index=today_df.index)
    if cal_tp is not None:
        p_tp = pd.Series(cal_tp.transform(p_tp), index=today_df.index)
    if cal_sl is not None:
        p_sl = pd.Series(cal_sl.transform(p_sl), index=today_df.index)

    today_df["p_tp"] = p_tp.values
    today_df["p_sl"] = p_sl.values
    today_df["score"] = compute_score(today_df["p_tp"], today_df["p_sl"])

    th = mv.threshold
    cand = today_df.sort_values("score", ascending=False).set_index("ticker")[["score"]]
    cand = cand[cand["score"] > th]

    returns_full = precompute_returns(close_by_ticker)
    corr = window_corr(returns_full, last_date, pcfg["correlation"]["window"])
    sel = greedy_filter(cand, corr,
                        threshold=pcfg["correlation"]["threshold"],
                        n_max=mv.n_star)
    log.info("[signal] candidates=%d selected=%d", len(cand), len(sel))

    weights = equal_weights(sel, exposure)

    report = {
        "date": str(last_date.date()),
        "regime": active,
        "universe": "stocks",
        "market_breadth": exposure,
        "n_star": int(mv.n_star),
        "model_version": mv.version,
        "model_retrain_date": mv.retrain_date,
        "selected": [],
        "cash": 1.0 - float(weights.sum()),
        "total_exposure": float(weights.sum()),
    }
    for t in sel:
        row = today_df[today_df["ticker"] == t].iloc[0]
        report["selected"].append({
            "ticker": t,
            "p_tp": float(row["p_tp"]),
            "p_sl": float(row["p_sl"]),
            "score": float(row["score"]),
            "target_weight": float(weights.get(t, 0.0)),
        })
    return report


def save_report(report: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"signal_{report['date']}.json"
    p.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return p