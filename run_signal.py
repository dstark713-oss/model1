
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
    path = Path(p)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load(path: str) -> dict:
    return yaml.safe_load(_resolve(path).read_text(encoding="utf-8"))


def _storage(cfg: dict) -> SqliteStorage:
    return SqliteStorage(
        StorageConfig(
            db_path=_resolve(cfg["data"]["db_path"]),
            artifacts_dir=PROJECT_ROOT / "artifacts",
        )
    )


def _print_buy_list(report: dict) -> None:
    print("\n" + "=" * 78)
    print("BUY LIST")
    print("=" * 78)

    buy_list = report["buy_list"]

    if not buy_list:
        print("  No selected stocks.")
        return

    print(
        f"{'rank':>5s} "
        f"{'ticker':10s} "
        f"{'p_tp':>9s} "
        f"{'p_sl':>9s} "
        f"{'score':>9s} "
        f"{'weight':>10s}"
    )

    print("-" * 78)

    for item in buy_list:
        print(
            f"{item['score_rank']:5d} "
            f"{item['ticker']:10s} "
            f"{item['p_tp']:9.4f} "
            f"{item['p_sl']:9.4f} "
            f"{item['score']:9.4f} "
            f"{item['target_weight']:10.4f}"
        )


def _print_score_ranking(report: dict) -> None:
    print("\n" + "=" * 100)
    print(f"SCORE RANKING — TOP {report['score_ranking_top_n']}")
    print("=" * 100)

    ranking = report["score_ranking"]

    if not ranking:
        print("  No valid model predictions.")
        return

    print(
        f"{'rank':>5s} "
        f"{'ticker':10s} "
        f"{'p_tp':>9s} "
        f"{'p_sl':>9s} "
        f"{'score':>9s} "
        f"{'threshold':>11s} "
        f"{'selected':>10s} "
        f"{'weight':>10s}"
    )

    print("-" * 100)

    for item in ranking:
        threshold_status = "YES" if item["threshold_pass"] else "NO"
        selected_status = "YES" if item["selected"] else "NO"

        print(
            f"{item['rank']:5d} "
            f"{item['ticker']:10s} "
            f"{item['p_tp']:9.4f} "
            f"{item['p_sl']:9.4f} "
            f"{item['score']:9.4f} "
            f"{threshold_status:>11s} "
            f"{selected_status:>10s} "
            f"{item['target_weight']:10.4f}"
        )


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

        res = update_to_today(
            st,
            sleep_between=scfg.get("sleep_between", 0.25),
        )

        print(f"[update] {json.dumps(res, ensure_ascii=False)}")

    report = build_signal(
        st,
        cfg,
        fcfg,
        mcfg,
        pcfg,
    )

    print("\n" + "=" * 78)
    print(
        f"SIGNAL  {report['date']}  "
        f"regime={report['regime']}  "
        f"breadth={report['market_breadth']:.3f}  "
        f"n*={report['n_star']}  "
        f"model={report.get('model_version', '?')}"
    )
    print("=" * 78)

    _print_buy_list(report)
    _print_score_ranking(report)

    print("\n" + "=" * 78)
    print("PORTFOLIO SUMMARY")
    print("=" * 78)

    print(
        f"Scored universe: {report['scored_universe_size']} | "
        f"Threshold passed: {report['threshold_pass_count']} | "
        f"Selected: {report['selected_count']}"
    )

    print(
        f"Cash: {report['cash']:.4f} | "
        f"Exposure: {report['total_exposure']:.4f}"
    )

    print(
        f"Score definition: {report['score_definition']} | "
        f"Threshold: {report['score_threshold']:.4f}"
    )

    path = save_report(
        report,
        PROJECT_ROOT / scfg["report_dir"],
    )

    print(f"\n[signal] saved -> {path}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="signal")

    parser.add_argument(
        "--config",
        default="configs/universe.yaml",
    )
    parser.add_argument(
        "--features-config",
        default="configs/features.yaml",
    )
    parser.add_argument(
        "--models-config",
        default="configs/models.yaml",
    )
    parser.add_argument(
        "--portfolio-config",
        default="configs/portfolio.yaml",
    )
    parser.add_argument(
        "--signal-config",
        default="configs/signal.yaml",
    )
    parser.add_argument(
        "--skip-update",
        action="store_true",
    )

    sub = parser.add_subparsers(dest="cmd")

    signal_parser = sub.add_parser(
        "run",
        help="Апдейт до сегодня + сигнал на сегодня",
    )
    signal_parser.set_defaults(func=cmd_signal)

    return parser


def main(argv=None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = build_parser()
    args = parser.parse_args(argv)

    if not getattr(args, "cmd", None):
        parser.print_help()
        return 0

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
