from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
import yaml

from backtest.engine import run_backtest
from backtest.metrics import summarize
from data.storage import SqliteStorage, StorageConfig
from features.pipeline import compute_features
from models.persistence import load_latest
from portfolio.breadth import breadth_ratio
from portfolio.correlation import precompute_returns, window_corr, greedy_filter
from portfolio.ranking import compute_score

PROJECT_ROOT = Path(__file__).resolve().parent


def _resolve(p: str) -> Path:
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _load(path: str) -> dict:
    return yaml.safe_load(_resolve(path).read_text())


def _storage(cfg: dict) -> SqliteStorage:
    return SqliteStorage(StorageConfig(
        db_path=_resolve(cfg["data"]["db_path"]),
        artifacts_dir=PROJECT_ROOT / "artifacts",
    ))


def cmd_run(args) -> int:
    cfg = _load(args.config)
    fcfg = _load(args.features_config)
    mcfg = _load(args.models_config)
    pcfg = _load(args.portfolio_config)
    bcfg = _load(args.backtest_config)["backtest"]

    loaded = load_latest(_resolve(mcfg["persistence"]["models_dir"]))
    if loaded is None:
        print("[live] no model version")
        return 1
    mv, tp_model, sl_model, cal_tp, cal_sl = loaded
    print(f"[live] model={mv.version} retrain={mv.retrain_date} n_star={mv.n_star}")

    st = _storage(cfg)

    # универс: TOP100 на последнюю дату (или на retrain_date)
    top = pd.read_parquet(_resolve("artifacts/phase2/top100_pit.parquet"))
    last_date = pd.Timestamp.now().normalize()
    top_recent = top[top["date"] <= last_date]
    tickers = sorted(top_recent.loc[top_recent["date"] == top_recent["date"].max(),
                                     "ticker"].tolist())

    # загрузка данных
    close_by_ticker, open_by_ticker = {}, {}
    for t in tickers:
        df = st.load(t)
        if df.empty:
            continue
        s = df.set_index("date")
        close_by_ticker[t] = s["close"]
        open_by_ticker[t] = s["open"]

    # precompute features по каждому тикеру, ограничиваем [retrain_date, last_date]
    retrain_dt = pd.Timestamp(mv.retrain_date)
    rows = []
    for t in tickers:
        px = st.load(t)
        if px.empty or len(px) < 300:
            continue
        feats = compute_features(px, fcfg)
        feats = feats.loc[(feats.index > retrain_dt) & (feats.index <= last_date)]
        if feats.empty:
            continue
        rows.append(feats.assign(ticker=t).reset_index())

    if not rows:
        print("[live] no rows after retrain_date")
        return 1
    live = pd.concat(rows, ignore_index=True)
    live["date"] = pd.to_datetime(live["date"])
    cols = mv.feature_names
    live = live.dropna(subset=cols, how="any")

    # предсказания зафиксированной моделью
    X = live[cols]
    p_tp = tp_model.predict_proba(X)[:, 1]
    p_sl = sl_model.predict_proba(X)[:, 1]
    if cal_tp is not None:
        p_tp = cal_tp.transform(pd.Series(p_tp)).values
    if cal_sl is not None:
        p_sl = cal_sl.transform(pd.Series(p_sl)).values
    live["p_tp"] = p_tp
    live["p_sl"] = p_sl
    live["score"] = compute_score(live["p_tp"], live["p_sl"])

    # бэктест по этим предсказаниям (N* и threshold — из версии модели)
    params_by_period = {(retrain_dt, last_date): {"n": mv.n_star, "threshold": mv.threshold}}

    # breadth и returns — по всему универсу
    b = breadth_ratio(close_by_ticker, pcfg["allocation"]["breadth_ema"])
    returns_full = precompute_returns(close_by_ticker)

    # используем run_backtest, но передаём ему live-предсказания
    bt = run_backtest(
        preds=live[["date", "ticker", "p_tp", "p_sl"]].copy(),
        open_by_ticker=open_by_ticker,
        close_by_ticker=close_by_ticker,
        params_by_period=params_by_period,
        pcfg=pcfg,
        bcfg=bcfg,
    )
    if bt.empty:
        print("[live] no trades")
        return 1

    summary = summarize(bt)
    print("[live] summary (after retrain):")
    for k, v in summary.items():
        print(f"  {k:15s} {v}")

    print(f"[live] period: {bt.index.min().date()}..{bt.index.max().date()} "
          f"({len(bt)} days)")

    out_dir = PROJECT_ROOT / "artifacts" / "live"
    out_dir.mkdir(parents=True, exist_ok=True)
    bt.to_parquet(out_dir / "live_backtest.parquet")
    (out_dir / "live_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[live] saved -> {out_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="live_validate")
    p.add_argument("--config", default="configs/universe.yaml")
    p.add_argument("--features-config", default="configs/features.yaml")
    p.add_argument("--models-config", default="configs/models.yaml")
    p.add_argument("--portfolio-config", default="configs/portfolio.yaml")
    p.add_argument("--backtest-config", default="configs/backtest.yaml")
    sub = p.add_subparsers(dest="cmd")
    sp = sub.add_parser("run")
    sp.set_defaults(func=cmd_run)
    return p


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "cmd", None):
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())