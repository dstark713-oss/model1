from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from data.storage import SqliteStorage, StorageConfig
from features.pipeline import compute_features
from models.calibration import ProbabilityCalibrator
from models.common import feature_columns
from models.persistence import save_model_version
from models.sl_model import SLModel
from models.tp_model import TPModel
from portfolio.selection import select_n_on_train
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


def _build_dataset(st: SqliteStorage, tickers: list[str],
                   fcfg: dict, tcfg: dict) -> tuple[pd.DataFrame, list[str]]:
    rows = []
    for t in tickers:
        px = st.load(t)
        if px.empty or len(px) < 300:
            continue
        feats = compute_features(px, fcfg)
        targ = label_triple_barrier(px, tcfg)
        df = feats.join(targ, how="inner")
        df = df.dropna(subset=["tp_hit", "sl_hit"])
        df = df[~df["ambiguous"]]
        df["ticker"] = t
        df.index.name = "date"
        rows.append(df.reset_index())
    if not rows:
        raise SystemExit("no data")
    full = pd.concat(rows, ignore_index=True)
    full["date"] = pd.to_datetime(full["date"])
    cols = feature_columns(full)
    full = full.dropna(subset=cols, how="any")
    return full, cols


def cmd_retrain(args) -> int:
    cfg = _load(args.config)
    fcfg = _load(args.features_config)
    tcfg = _load(args.targets_config)["targets"]
    mcfg = _load(args.models_config)
    pcfg = _load(args.portfolio_config)

    retrain_date = args.date or datetime.now().strftime("%Y-%m-%d")

    top = pd.read_parquet(PROJECT_ROOT / "artifacts" / "phase2" / "top100_pit.parquet")
    top_until = top[top["date"] <= pd.Timestamp(retrain_date)]
    last_top_date = top_until["date"].max()
    tickers = top_until.loc[top_until["date"] == last_top_date, "ticker"].tolist()
    print(f"[retrain] date={retrain_date} universe={len(tickers)} (top100 at {last_top_date.date()})")

    st = _storage(cfg)
    full, cols = _build_dataset(st, tickers, fcfg, tcfg)
    full = full[full["date"] <= pd.Timestamp(retrain_date)]
    print(f"[retrain] dataset rows={len(full)} features={len(cols)}")

    X = full[cols]
    y_tp = full["tp_hit"].astype(int)
    y_sl = full["sl_hit"].astype(int)

    tp = TPModel(mcfg["models"]["tp"]).fit(X, y_tp)
    sl = SLModel(mcfg["models"]["sl"]).fit(X, y_sl)

    cut = int(len(full) * 0.8)
    cal_tp = ProbabilityCalibrator(mcfg["calibration"]["method"]).fit(
        tp.predict_proba(X.iloc[cut:]), y_tp.iloc[cut:])
    cal_sl = ProbabilityCalibrator(mcfg["calibration"]["method"]).fit(
        sl.predict_proba(X.iloc[cut:]), y_sl.iloc[cut:])

    n_star = select_n_on_train(full, cols, mcfg, pcfg)
    threshold = float(pcfg["selection"]["score_threshold"])

    models_dir = _resolve(mcfg["persistence"]["models_dir"])
    mv = save_model_version(
        models_dir=models_dir,
        retrain_date=retrain_date,
        tp_model=tp,
        sl_model=sl,
        calibrator_tp=cal_tp,
        calibrator_sl=cal_sl,
        feature_names=cols,
        n_star=n_star,
        threshold=threshold,
        train_rows=len(full),
        train_start=str(full["date"].min().date()),
        train_end=str(full["date"].max().date()),
        hyperparams_tp=mcfg["models"]["tp"],
        hyperparams_sl=mcfg["models"]["sl"],
        calibration_method=mcfg["calibration"]["method"],
    )
    print(f"[retrain] saved version {mv.version} -> {models_dir / mv.version}")
    print(f"[retrain] n_star={mv.n_star} threshold={mv.threshold} rows={mv.train_rows}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="retrain")
    p.add_argument("--config", default="configs/universe.yaml")
    p.add_argument("--features-config", default="configs/features.yaml")
    p.add_argument("--targets-config", default="configs/targets.yaml")
    p.add_argument("--models-config", default="configs/models.yaml")
    p.add_argument("--portfolio-config", default="configs/portfolio.yaml")
    p.add_argument("--date", default=None, help="Дата retrain (YYYY-MM-DD), по умолчанию сегодня")
    sub = p.add_subparsers(dest="cmd")
    sp = sub.add_parser("run")
    sp.set_defaults(func=cmd_retrain)
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