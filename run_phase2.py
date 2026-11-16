# run_phase2.py
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

from data.storage import SqliteStorage, StorageConfig
from portfolio.regime import compute_regime
from portfolio.universe import compute_top100

import pandas as pd
from features.pipeline import compute_features
from features.market import compute_breadth, compute_position_vs_ema20

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


def cmd_regime(args) -> int:
    cfg = _load(args.config)
    st = _storage(cfg)
    imoex = st.load("IMOEX")
    r = compute_regime(imoex, ema_period=cfg["regime"]["ema_period"]).dropna()
    print(f"[regime] IMOEX rows={len(imoex)} regime rows={len(r)}")
    print(r.tail(10).to_string())
    flips = r[r != r.shift(1)].dropna()
    print(f"\n[regime] switches: {len(flips)}")
    print(flips.tail(10).to_string())
    out = PROJECT_ROOT / "artifacts" / "phase2" / "imoex_regime.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    return 0


def cmd_top100(args) -> int:
    cfg = _load(args.config)
    st = _storage(cfg)
    exclude = {e["local"] for e in cfg["moex"]["extra"]}
    exclude |= {y["local"] for y in cfg["yahoo"]["tickers"]}
    top = compute_top100(
        st,
        window=cfg["universe"]["top100_window"],
        top_n=cfg["universe"]["top_n"],
        exclude=exclude,
    )
    print(f"[top100] rows={len(top)} dates={top['date'].nunique()}")
    print(top.tail(20).to_string())
    out = PROJECT_ROOT / "artifacts" / "phase2" / "top100_pit.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    top.to_parquet(out, index=False)
    print(f"[top100] saved -> {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="phase2")
    p.add_argument("--config", default="configs/universe.yaml")
    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("regime", help="IMOEX regime по Close vs EMA").set_defaults(func=cmd_regime)
    sub.add_parser("top100", help="Point-in-time TOP100 по value").set_defaults(func=cmd_top100)
    sp = sub.add_parser("features", help="Фичи на одном тикере (smoke)")
    sp.add_argument("ticker")
    sp.add_argument("--features-config", default="configs/features.yaml")
    sp.set_defaults(func=cmd_features)

    sp = sub.add_parser("breadth", help="Market Breadth по активному универсу")
    sp.add_argument("--features-config", default="configs/features.yaml")
    sp.set_defaults(func=cmd_breadth)
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

def _load_features_cfg(path: str = "configs/features.yaml") -> dict:
    return yaml.safe_load(_resolve(path).read_text())


def cmd_features(args) -> int:
    cfg = _load(args.config)
    fcfg = _load_features_cfg(args.features_config)
    st = _storage(cfg)
    df = st.load(args.ticker)
    if df.empty:
        print(f"[features] {args.ticker}: no data")
        return 1
    out = compute_features(df, fcfg)
    print(f"[features] {args.ticker}: rows={len(out)} cols={len(out.columns)}")
    print("columns:", list(out.columns))
    out_dir = PROJECT_ROOT / "artifacts" / "phase2" / "features"
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{args.ticker}.parquet"
    out.to_parquet(p, index=False)
    print(f"[features] saved -> {p}")
    return 0


def cmd_breadth(args) -> int:
    cfg = _load(args.config)
    fcfg = _load_features_cfg(args.features_config)
    st = _storage(cfg)
    ema20 = fcfg["market"]["ema20_period"]

    # активный универс на последнюю дату
    regime = compute_regime(st.load("IMOEX"),
                            ema_period=cfg["regime"]["ema_period"]).dropna()
    last_date = regime.index.max()
    active = regime.loc[last_date]
    print(f"[breadth] last_date={last_date.date()} regime={active}")

    if active == "stocks":
        # point-in-time TOP100 на последнюю дату
        top = pd.read_parquet(PROJECT_ROOT / "artifacts" / "phase2" / "top100_pit.parquet")
        tickers = top.loc[top["date"] == last_date, "ticker"].tolist()
    else:
        tickers = [y["local"] for y in cfg["yahoo"]["tickers"]]
        tickers += [e["local"] for e in cfg["moex"]["extra"]
                    if e["local"] not in ("IMOEX", "RGBI")]

    print(f"[breadth] universe size={len(tickers)}")
    prices = {t: st.load(t)[["date", "close"]] for t in tickers}
    prices = {t: df for t, df in prices.items() if not df.empty}
    print(f"[breadth] with data={len(prices)}")

    b = compute_breadth(prices, ema_period=ema20)
    print(f"[breadth] rows={len(b)}")
    print(b.tail(10).to_string())

    out_dir = PROJECT_ROOT / "artifacts" / "phase2"
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "breadth.csv"
    b.to_csv(p)
    print(f"[breadth] saved -> {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())