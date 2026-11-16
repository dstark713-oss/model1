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
    pcfg = _load(args.portfolio_config)
    bcfg = _load(args.backtest_config)["backtest"]

    preds = pd.read_parquet(PROJECT_ROOT / "artifacts" / "phase5" / "oos_predictions.parquet")
    wf = json.loads((PROJECT_ROOT / "artifacts" / "phase5" / "wf_results.json").read_text())

    params_by_period = {}
    for r in wf:
        if r.get("skipped"):
            continue
        ts = pd.Timestamp(r["test_start"])
        te = pd.Timestamp(r["test_end"])
        params_by_period[(ts, te)] = {
            "n": int(r["n_star"]),
            "threshold": float(r.get("threshold_star", 0.0)),
        }

    st = _storage(cfg)
    tickers = sorted(preds["ticker"].unique())
    open_by_ticker, close_by_ticker = {}, {}
    for t in tickers:
        df = st.load(t)
        if df.empty:
            continue
        s = df.set_index("date")
        open_by_ticker[t] = s["open"]
        close_by_ticker[t] = s["close"]

    bt = run_backtest(preds, open_by_ticker, close_by_ticker,
                      params_by_period, pcfg, bcfg)
    if bt.empty:
        print("[backtest] no results")
        return 1

    summary = summarize(bt)
    print("[backtest] summary:")
    for k, v in summary.items():
        print(f"  {k:15s} {v}")

    out_dir = PROJECT_ROOT / "artifacts" / "phase7"
    out_dir.mkdir(parents=True, exist_ok=True)
    bt.to_parquet(out_dir / "backtest_daily.parquet")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[backtest] saved -> {out_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="phase7")
    p.add_argument("--config", default="configs/universe.yaml")
    p.add_argument("--portfolio-config", default="configs/portfolio.yaml")
    p.add_argument("--backtest-config", default="configs/backtest.yaml")
    sub = p.add_subparsers(dest="cmd")
    sp = sub.add_parser("run", help="Бэктест stocks")
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