from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from data.storage import SqliteStorage, StorageConfig
from portfolio.breadth import breadth_ratio
from portfolio.correlation import build_returns_matrix
from portfolio.ranking import compute_score, rank_candidates
from portfolio.selection import choose_n_assets

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


def cmd_demo(args) -> int:
    cfg = _load(args.config)
    pcfg = _load(args.portfolio_config)
    st = _storage(cfg)

    # demo: последняя дата из top100, фиктивные p_tp/p_sl (пока без моделей в проде)
    top = pd.read_parquet(PROJECT_ROOT / "artifacts" / "phase2" / "top100_pit.parquet")
    last_date = top["date"].max()
    tickers = top.loc[top["date"] == last_date, "ticker"].tolist()

    px = {t: st.load(t).set_index("date")["close"] for t in tickers}
    px = {t: s for t, s in px.items() if not s.empty}

    b = breadth_ratio(px, pcfg["allocation"]["breadth_ema"]).dropna()
    exposure = float(b.iloc[-1]) if not b.empty else 0.0
    print(f"[phase6] date={last_date.date()} breadth={exposure:.3f} universe={len(px)}")

    # заглушка: p_tp=0.5, p_sl=0.5 -> score=0.25, ранжирование не работает,
    # но демонстрирует пайплайн. Реальные p_tp/p_sl придут из WF-моделей.
    p_tp = pd.Series(0.5, index=list(px.keys()))
    p_sl = pd.Series(0.5, index=list(px.keys()))
    score = compute_score(p_tp, p_sl)
    cand = rank_candidates(pd.DataFrame({"score": score}),
                           threshold=pcfg["selection"]["score_threshold"])
    print(f"[phase6] candidates: {len(cand)}")

    returns_window = build_returns_matrix(px, pcfg["correlation"]["window"])

    def metric_fn(sel, w):
        return float(score.reindex(sel).fillna(0).mul(w).sum())

    n, sel, w = choose_n_assets(cand, pd.Series(1.0, index=cand.index),
                                returns_window, pcfg, exposure, metric_fn)
    print(f"[phase6] N*={n} selected={sel}")
    print(f"[phase6] weights:\n{w.to_string()}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="phase6")
    p.add_argument("--config", default="configs/universe.yaml")
    p.add_argument("--portfolio-config", default="configs/portfolio.yaml")
    sub = p.add_subparsers(dest="cmd")
    sp = sub.add_parser("demo", help="Пайплайн ranking/corr/allocation на последней дате")
    sp.set_defaults(func=cmd_demo)
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