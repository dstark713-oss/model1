from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

from data.storage import SqliteStorage, StorageConfig
from signals.daily import build_signal, save_report
from signals.updater import update_to_today

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


def cmd_signal(args) -> int:
    cfg = _load(args.config)
    fcfg = _load(args.features_config)
    mcfg = _load(args.models_config)
    pcfg = _load(args.portfolio_config)
    scfg = _load(args.signal_config)["signal"]

    st = _storage(cfg)

    if not args.skip_update:
        log = logging.getLogger("signal")
        log.info("[signal] update to today ...")
        res = update_to_today(st, sleep_between=scfg.get("sleep_between", 0.25))
        print(f"[update] {json.dumps(res, ensure_ascii=False)}")

    report = build_signal(st, cfg, fcfg, mcfg, pcfg)

    print("\n" + "=" * 60)
    print(f"SIGNAL  {report['date']}  regime={report['regime']}  "
          f"breadth={report['market_breadth']:.3f}  n*={report['n_star']}  "
          f"model={report.get('model_version','?')}")
    print("=" * 60)
    if report["selected"]:
        print(f"{'ticker':10s} {'p_tp':>8s} {'p_sl':>8s} {'score':>8s} {'weight':>10s}")
        for s in report["selected"]:
            print(f"{s['ticker']:10s} {s['p_tp']:8.4f} {s['p_sl']:8.4f} "
                  f"{s['score']:8.4f} {s['target_weight']:10.4f}")
    else:
        print("  (no candidates)")
    print(f"  cash: {report['cash']:.4f}  exposure: {report['total_exposure']:.4f}")

    p = save_report(report, PROJECT_ROOT / scfg["report_dir"])
    print(f"\n[signal] saved -> {p}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="signal")
    p.add_argument("--config", default="configs/universe.yaml")
    p.add_argument("--features-config", default="configs/features.yaml")
    p.add_argument("--models-config", default="configs/models.yaml")
    p.add_argument("--portfolio-config", default="configs/portfolio.yaml")
    p.add_argument("--signal-config", default="configs/signal.yaml")
    p.add_argument("--skip-update", action="store_true")
    sub = p.add_subparsers(dest="cmd")
    sp = sub.add_parser("run", help="Апдейт до сегодня + сигнал на сегодня")
    sp.set_defaults(func=cmd_signal)
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