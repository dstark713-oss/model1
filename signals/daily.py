
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from data.storage import SqliteStorage
from features.pipeline import compute_features
from models.persistence import load_latest
from portfolio.allocation import equal_weights
from portfolio.breadth import breadth_ratio
from portfolio.correlation import (
    precompute_returns,
    window_corr,
    greedy_filter,
)
from portfolio.ranking import compute_score
from portfolio.regime import compute_regime


log = logging.getLogger(__name__)

# Сколько бумаг показываем в отдельном score-рейтинге.
SCORE_RANKING_TOP_N = 15


def _resolve(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else Path(__file__).resolve().parent.parent / path


def _build_today_features(
    st: SqliteStorage,
    tickers: list[str],
    fcfg: dict,
    last_date: pd.Timestamp,
) -> pd.DataFrame:
    rows = []

    for ticker in tickers:
        px = st.load(ticker)

        if px.empty or len(px) < 300:
            continue

        feats = compute_features(px, fcfg)

        if feats.empty:
            continue

        sub = feats.loc[feats.index <= last_date]

        if sub.empty:
            continue

        row = sub.loc[[sub.index.max()]].copy()
        row["ticker"] = ticker
        row.index.name = "date"

        rows.append(row.reset_index())

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


def build_signal(
    st: SqliteStorage,
    cfg: dict,
    fcfg: dict,
    mcfg: dict,
    pcfg: dict,
) -> dict:
    loaded = load_latest(_resolve(mcfg["persistence"]["models_dir"]))

    if loaded is None:
        raise SystemExit("no model version found, run retrain.py first")

    mv, tp_model, sl_model, cal_tp, cal_sl = loaded

    log.info(
        "[signal] model version=%s retrain_date=%s",
        mv.version,
        mv.retrain_date,
    )

    # ---------------------------------------------------------
    # 1. Market regime and point-in-time universe
    # ---------------------------------------------------------

    imoex = st.load("IMOEX")

    regime = compute_regime(
        imoex,
        ema_period=cfg["regime"]["ema_period"],
    ).dropna()

    top = pd.read_parquet(_resolve("artifacts/phase2/top100_pit.parquet"))

    top_last = top["date"].max()
    tickers = top.loc[top["date"] == top_last, "ticker"].tolist()

    # Общая последняя торговая дата: есть и режим, и TOP-100.
    last_date = min(regime.index.max(), top_last)
    active_regime = str(regime.loc[last_date])

    log.info(
        "[signal] last_date=%s regime=%s",
        last_date.date(),
        active_regime,
    )
    log.info(
        "[signal] universe=stocks tickers=%d (%s)",
        len(tickers),
        top_last.date(),
    )

    # ---------------------------------------------------------
    # 2. Market breadth determines total exposure
    # ---------------------------------------------------------

    close_by_ticker = {}

    for ticker in tickers:
        df = st.load(ticker)

        if df.empty:
            continue

        close_by_ticker[ticker] = df.set_index("date")["close"]

    breadth = breadth_ratio(
        close_by_ticker,
        pcfg["allocation"]["breadth_ema"],
    ).dropna()

    exposure = float(breadth.get(last_date, 0.0)) if not breadth.empty else 0.0

    log.info("[signal] breadth=%.3f", exposure)

    # ---------------------------------------------------------
    # 3. Build features and calculate model scores
    # ---------------------------------------------------------

    today_df = _build_today_features(
        st,
        tickers,
        fcfg,
        last_date,
    )

    if today_df.empty:
        raise SystemExit("no features on last date")

    feature_cols = mv.feature_names

    today_df = today_df.dropna(subset=feature_cols, how="any").copy()

    if today_df.empty:
        raise SystemExit("no valid rows after feature filtering")


    X = today_df[feature_cols]

    # Calibrator.transform() expects pandas Series, not numpy arrays.
    p_tp = pd.Series(
        tp_model.predict_proba(X)[:, 1],
        index=today_df.index,
        name="p_tp",
    )
    p_sl = pd.Series(
        sl_model.predict_proba(X)[:, 1],
        index=today_df.index,
        name="p_sl",
    )

    if cal_tp is not None:
        p_tp = cal_tp.transform(p_tp)

    if cal_sl is not None:
        p_sl = cal_sl.transform(p_sl)

    today_df["p_tp"] = p_tp
    today_df["p_sl"] = p_sl
    today_df["score"] = compute_score(
        today_df["p_tp"],
        today_df["p_sl"],
    )

    today_df = today_df.replace(
        [float("inf"), float("-inf")],
        pd.NA,
    ).dropna(subset=["p_tp", "p_sl", "score"])

    if today_df.empty:
        raise SystemExit("no valid model predictions")

    # ---------------------------------------------------------
    # 4. Full score ranking
    #
    # Ranking is built BEFORE threshold and correlation filters.
    # This lets us see the strongest model candidates even if
    # they do not enter the final portfolio.
    # ---------------------------------------------------------

    threshold = float(mv.threshold)

    ranked = (
        today_df[
            ["ticker", "p_tp", "p_sl", "score"]
        ]
        .sort_values(
            ["score", "ticker"],
            ascending=[False, True],
        )
        .reset_index(drop=True)
    )

    ranked["score_rank"] = ranked.index + 1
    ranked["threshold_pass"] = ranked["score"] > threshold

    # ---------------------------------------------------------
    # 5. Portfolio selection
    #
    # Existing selection mechanics remain unchanged:
    # threshold -> correlation filter -> N* -> equal weights.
    # ---------------------------------------------------------

    candidates = (
        today_df.loc[
            today_df["score"] > threshold,
            ["ticker", "score"],
        ]
        .sort_values(
            ["score", "ticker"],
            ascending=[False, True],
        )
        .set_index("ticker")
    )

    returns_full = precompute_returns(close_by_ticker)

    corr = window_corr(
        returns_full,
        last_date,
        pcfg["correlation"]["window"],
    )

    selected = greedy_filter(
        candidates,
        corr,
        threshold=pcfg["correlation"]["threshold"],
        n_max=mv.n_star,
    )

    weights = equal_weights(selected, exposure)

    selected_set = set(selected)
    weight_map = weights.to_dict()

    rank_map = dict(
        zip(ranked["ticker"], ranked["score_rank"])
    )

    log.info(
        "[signal] scored=%d threshold_pass=%d selected=%d",
        len(ranked),
        len(candidates),
        len(selected),
    )

    # ---------------------------------------------------------
    # 6. Build two independent output sections
    # ---------------------------------------------------------

    buy_list = []

    for ticker in selected:
        row = today_df.loc[today_df["ticker"] == ticker].iloc[0]

        buy_list.append({
            "score_rank": int(rank_map[ticker]),
            "ticker": ticker,
            "p_tp": float(row["p_tp"]),
            "p_sl": float(row["p_sl"]),
            "score": float(row["score"]),
            "target_weight": float(weight_map.get(ticker, 0.0)),
        })

    score_ranking = []

    for row in ranked.head(SCORE_RANKING_TOP_N).itertuples(index=False):
        ticker = row.ticker

        score_ranking.append({
            "rank": int(row.score_rank),
            "ticker": ticker,
            "p_tp": float(row.p_tp),
            "p_sl": float(row.p_sl),
            "score": float(row.score),
            "threshold_pass": bool(row.threshold_pass),
            "selected": ticker in selected_set,
            "target_weight": float(weight_map.get(ticker, 0.0)),
        })

    total_exposure = float(weights.sum())

    report = {
        "date": str(last_date.date()),
        "regime": active_regime,
        "universe": "stocks",
        "market_breadth": exposure,
        "n_star": int(mv.n_star),
        "model_version": mv.version,
        "model_retrain_date": mv.retrain_date,
        "score_definition": "p_tp * (1 - p_sl)",
        "score_threshold": threshold,
        "score_ranking_top_n": SCORE_RANKING_TOP_N,
        "scored_universe_size": int(len(ranked)),
        "threshold_pass_count": int(len(candidates)),
        "selected_count": int(len(selected)),
        "buy_list": buy_list,
        "score_ranking": score_ranking,

        # Backward compatibility for existing consumers.
        "selected": buy_list,

        "cash": 1.0 - total_exposure,
        "total_exposure": total_exposure,
    }

    return report


def save_report(report: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    path = out_dir / f"signal_{report['date']}.json"

    path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return path
