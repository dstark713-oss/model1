from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import yaml

from data.storage import SqliteStorage, StorageConfig
from targets.triple_barrier import label_triple_barrier

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


def cmd_targets(args) -> int:
    cfg = _load(args.config)
    tcfg = _load(args.targets_config)["targets"]
    st = _storage(cfg)
    df = st.load(args.ticker)
    if df.empty:
        print(f"[targets] {args.ticker}: no data")
        return 1
    out = label_triple_barrier(df, tcfg)
    valid = out.dropna(subset=["tp_hit", "sl_hit"])
    amb = int(out["ambiguous"].sum())
    to = int(out["timeout"].sum())
    tp = int((valid["tp_hit"] == 1).sum())
    sl = int((valid["sl_hit"] == 1).sum())
    print(f"[targets] {args.ticker}: H={tcfg['h']} rows={len(out)}")
    print(f"  valid={len(valid)} tp_first={tp} sl_first={sl} "
          f"timeout={to} ambiguous={amb}")
    print(valid.tail(10).to_string())
    out_dir = PROJECT_ROOT / "artifacts" / "phase3"
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{args.ticker}_targets.parquet"
    out.to_parquet(p)
    print(f"[targets] saved -> {p}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="phase3")
    p.add_argument("--config", default="configs/universe.yaml")
    p.add_argument("--targets-config", default="configs/targets.yaml")
    sub = p.add_subparsers(dest="cmd")
    sp = sub.add_parser("targets", help="Triple Barrier разметка на тикере")
    sp.add_argument("ticker")
    sp.set_defaults(func=cmd_targets)
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