from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
import yaml

from data.storage import SqliteStorage, StorageConfig
from features.pipeline import compute_features
from models.calibration import ProbabilityCalibrator
from models.common import feature_columns, time_split
from models.metrics import classification_metrics, calibration_table
from models.sl_model import SLModel
from models.tp_model import TPModel
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


def build_dataset(ticker: str, cfg: dict, fcfg: dict, tcfg: dict) -> pd.DataFrame:
    st = _storage(cfg)
    px = st.load(ticker)
    if px.empty:
        raise SystemExit(f"{ticker}: no data")
    feats = compute_features(px, fcfg)
    targets = label_triple_barrier(px, tcfg)
    df = feats.join(targets, how="inner")
    df = df.dropna(subset=["tp_hit", "sl_hit"])
    df = df[~df["ambiguous"]]
    return df


def cmd_smoke(args) -> int:
    cfg = _load(args.config)
    fcfg = _load(args.features_config)
    tcfg = _load(args.targets_config)["targets"]
    mcfg = _load(args.models_config)

    df = build_dataset(args.ticker, cfg, fcfg, tcfg)
    df = df.sort_index()
    cols = feature_columns(df)
    df = df.dropna(subset=cols, how="any")  # прогрев фич
    print(f"[phase4] {args.ticker}: dataset rows={len(df)} features={len(cols)}")

    tr, te = time_split(len(df), mcfg["holdout"]["test_fraction"],
                        mcfg["holdout"]["embargo_days"])
    train, test = df.iloc[tr], df.iloc[te]
    print(f"[phase4] train={len(train)} test={len(test)} "
          f"train_end={train.index.max().date()} test_start={test.index.min().date()}")

    Xtr, Xte = train[cols], test[cols]
    results = {}

    for name, ModelCls, target in [
        ("tp", TPModel, "tp_hit"),
        ("sl", SLModel, "sl_hit"),
    ]:
        ytr = train[target].astype(int)
        yte = test[target].astype(int)

        model = ModelCls(mcfg["models"][name]).fit(Xtr, ytr)
        p_te_raw = model.predict_proba(Xte)

        # калибровка на хвосте train (внутренняя validation)
        cal_frac = 0.2
        cut = int(len(train) * (1 - cal_frac))
        cal_X = Xtr.iloc[cut:]
        cal_y = ytr.iloc[cut:]
        p_cal_raw = model.predict_proba(cal_X)
        calibrator = ProbabilityCalibrator(mcfg["calibration"]["method"]).fit(p_cal_raw, cal_y)
        p_te = calibrator.transform(p_te_raw)

        m_raw = classification_metrics(yte, p_te_raw)
        m_cal = classification_metrics(yte, p_te)
        results[name] = {"raw": m_raw, "calibrated": m_cal}
        print(f"\n[{name.upper()}] test metrics")
        print(f"  raw        : {m_raw}")
        print(f"  calibrated : {m_cal}")
        print("  calibration table (calibrated):")
        print(calibration_table(yte, p_te).to_string(index=False))

    out_dir = PROJECT_ROOT / "artifacts" / "phase4"
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{args.ticker}_smoke.json"
    p.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\n[phase4] saved -> {p}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="phase4")
    p.add_argument("--config", default="configs/universe.yaml")
    p.add_argument("--features-config", default="configs/features.yaml")
    p.add_argument("--targets-config", default="configs/targets.yaml")
    p.add_argument("--models-config", default="configs/models.yaml")
    sub = p.add_subparsers(dest="cmd")
    sp = sub.add_parser("smoke", help="Обучение TP/SL на одном тикере, holdout")
    sp.add_argument("ticker")
    sp.set_defaults(func=cmd_smoke)
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