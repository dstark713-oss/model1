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
from features.pipeline import compute_features
from models.calibration import ProbabilityCalibrator
from models.common import feature_columns
from models.metrics import classification_metrics
from models.sl_model import SLModel
from models.tp_model import TPModel
from portfolio.selection import select_n_on_train
from targets.triple_barrier import label_triple_barrier
from validation.walk_forward import generate_folds, WFFold

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


def build_universe_dataset(cfg: dict, fcfg: dict, tcfg: dict) -> pd.DataFrame:
    st = _storage(cfg)
    top = pd.read_parquet(PROJECT_ROOT / "artifacts" / "phase2" / "top100_pit.parquet")
    top = top.drop_duplicates(subset=["date", "ticker"])
    tickers = sorted(top["ticker"].unique())
    print(f"[wf] universe=stocks: {len(tickers)} tickers, "
          f"{top['date'].nunique()} dates")

    rows = []
    for t in tickers:
        px = st.load(t)
        if px.empty or len(px) < 300:
            continue
        try:
            feats = compute_features(px, fcfg)
            targ = label_triple_barrier(px, tcfg)
        except Exception as e:
            print(f"[wf] {t}: skip ({e})")
            continue
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
    full = full.merge(top, on=["date", "ticker"], how="inner")
    return full


def run_fold(fold: WFFold, full: pd.DataFrame, cols: list[str],
             mcfg: dict, pcfg: dict) -> dict:
    df = full.sort_values("date")
    train_mask = (df["date"] >= fold.train_start) & (df["date"] <= fold.train_end)
    test_mask = (df["date"] >= fold.test_start) & (df["date"] <= fold.test_end)

    train = df[train_mask].copy()
    test = df[test_mask].copy()
    if len(train) == 0 or len(test) == 0:
        return {"fold_id": fold.fold_id, "skipped": True}

    Xtr, Xte = train[cols], test[cols]

    preds = {}
    metrics = {}
    for name, ModelCls, target in [("tp", TPModel, "tp_hit"),
                                   ("sl", SLModel, "sl_hit")]:
        ytr = train[target].astype(int)
        yte = test[target].astype(int)
        model = ModelCls(mcfg["models"][name]).fit(Xtr, ytr)
        p_te = model.predict_proba(Xte)

        cut = int(len(train) * 0.8)
        cal_X, cal_y = Xtr.iloc[cut:], ytr.iloc[cut:]
        p_cal = model.predict_proba(cal_X)
        cal = ProbabilityCalibrator(mcfg["calibration"]["method"]).fit(p_cal, cal_y)
        p_te_cal = cal.transform(p_te)

        metrics[name] = classification_metrics(yte, p_te_cal)
        preds[name] = pd.DataFrame({
            "date": test["date"].values,
            "ticker": test["ticker"].values,
            f"p_{name}": p_te_cal.values,
        })

    pred_df = preds["tp"].merge(preds["sl"], on=["date", "ticker"], how="inner")

    n_star = select_n_on_train(train, cols, mcfg, pcfg)
    threshold_star = pcfg["selection"]["score_threshold"]

    return {
        "fold_id": fold.fold_id,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_start": str(fold.train_start.date()),
        "train_end": str(fold.train_end.date()),
        "test_start": str(fold.test_start.date()),
        "test_end": str(fold.test_end.date()),
        "n_star": int(n_star),
        "threshold_star": float(threshold_star),
        "tp": {"metrics": metrics["tp"]},
        "sl": {"metrics": metrics["sl"]},
        "preds": pred_df,
    }


def cmd_run(args) -> int:
    cfg = _load(args.config)
    fcfg = _load(args.features_config)
    tcfg = _load(args.targets_config)["targets"]
    mcfg = _load(args.models_config)
    wcfg = _load(args.wf_config)["wf"]
    pcfg = _load(args.portfolio_config)

    full = build_universe_dataset(cfg, fcfg, tcfg)
    print(f"[wf] total dataset rows={len(full)}")
    full["date"] = pd.to_datetime(full["date"])

    cols = feature_columns(full)
    full = full.dropna(subset=cols, how="any")
    print(f"[wf] after dropna rows={len(full)} features={len(cols)}")

    folds = generate_folds(pd.DatetimeIndex(full["date"].unique()), wcfg)
    print(f"[wf] folds={len(folds)}")

    results = []
    all_preds = []
    for f in folds:
        r = run_fold(f, full, cols, mcfg, pcfg)
        if r.get("skipped"):
            continue
        pred_df = r.pop("preds")
        pred_df["fold_id"] = f.fold_id
        all_preds.append(pred_df)
        results.append(r)
        print(f"[wf] fold {f.fold_id} done: n_star={r['n_star']} "
              f"tp_auc={r['tp']['metrics'].get('roc_auc'):.3f} "
              f"sl_auc={r['sl']['metrics'].get('roc_auc'):.3f}")

    out_dir = PROJECT_ROOT / "artifacts" / "phase5"
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "wf_results.json"
    p.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"[wf] saved -> {p}")

    if all_preds:
        preds_all = pd.concat(all_preds, ignore_index=True)
        pp = out_dir / "oos_predictions.parquet"
        preds_all.to_parquet(pp, index=False)
        print(f"[wf] oos preds -> {pp} ({len(preds_all)} rows)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="phase5")
    p.add_argument("--config", default="configs/universe.yaml")
    p.add_argument("--features-config", default="configs/features.yaml")
    p.add_argument("--targets-config", default="configs/targets.yaml")
    p.add_argument("--models-config", default="configs/models.yaml")
    p.add_argument("--wf-config", default="configs/wf.yaml")
    p.add_argument("--portfolio-config", default="configs/portfolio.yaml")
    sub = p.add_subparsers(dest="cmd")
    sp = sub.add_parser("run", help="Walk-Forward по stocks")
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