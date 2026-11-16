from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier

from models.calibration import ProbabilityCalibrator


@dataclass
class ModelVersion:
    version: str
    retrain_date: str
    created_at: str
    tp_path: str
    sl_path: str
    calibrator_tp_path: str | None
    calibrator_sl_path: str | None
    feature_names: list[str]
    n_star: int
    threshold: float
    train_rows: int
    train_start: str
    train_end: str
    hyperparams_tp: dict
    hyperparams_sl: dict
    calibration_method: str


def _version_dir(models_dir: Path, version: str) -> Path:
    d = models_dir / version
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_model_version(models_dir: Path,
                       retrain_date: str,
                       tp_model,
                       sl_model,
                       calibrator_tp: ProbabilityCalibrator | None,
                       calibrator_sl: ProbabilityCalibrator | None,
                       feature_names: list[str],
                       n_star: int,
                       threshold: float,
                       train_rows: int,
                       train_start: str,
                       train_end: str,
                       hyperparams_tp: dict,
                       hyperparams_sl: dict,
                       calibration_method: str) -> ModelVersion:
    version = f"v_{retrain_date}"
    d = _version_dir(models_dir, version)

    tp_path = d / "tp.cbm"
    sl_path = d / "sl.cbm"
    tp_model.model.save_model(str(tp_path))
    sl_model.model.save_model(str(sl_path))

    cal_tp_path, cal_sl_path = None, None
    if calibrator_tp is not None and calibrator_tp.model is not None:
        cal_tp_path = d / "cal_tp.pkl"
        import pickle
        with open(cal_tp_path, "wb") as f:
            pickle.dump(calibrator_tp, f)
    if calibrator_sl is not None and calibrator_sl.model is not None:
        cal_sl_path = d / "cal_sl.pkl"
        import pickle
        with open(cal_sl_path, "wb") as f:
            pickle.dump(calibrator_sl, f)

    mv = ModelVersion(
        version=version,
        retrain_date=retrain_date,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        tp_path=str(tp_path),
        sl_path=str(sl_path),
        calibrator_tp_path=str(cal_tp_path) if cal_tp_path else None,
        calibrator_sl_path=str(cal_sl_path) if cal_sl_path else None,
        feature_names=list(feature_names),
        n_star=int(n_star),
        threshold=float(threshold),
        train_rows=int(train_rows),
        train_start=train_start,
        train_end=train_end,
        hyperparams_tp=hyperparams_tp,
        hyperparams_sl=hyperparams_sl,
        calibration_method=calibration_method,
    )
    (d / "metadata.json").write_text(json.dumps(asdict(mv), indent=2, ensure_ascii=False))

    latest = models_dir / "latest.json"
    latest.write_text(json.dumps({"version": version}, indent=2))
    return mv


def load_latest(models_dir: Path) -> tuple[ModelVersion, object, object,
                                           ProbabilityCalibrator | None,
                                           ProbabilityCalibrator | None] | None:
    latest = models_dir / "latest.json"
    if not latest.exists():
        return None
    version = json.loads(latest.read_text())["version"]
    d = models_dir / version
    meta = json.loads((d / "metadata.json").read_text())
    mv = ModelVersion(**meta)

    tp = CatBoostClassifier()
    tp.load_model(mv.tp_path)
    sl = CatBoostClassifier()
    sl.load_model(mv.sl_path)

    cal_tp, cal_sl = None, None
    if mv.calibrator_tp_path and Path(mv.calibrator_tp_path).exists():
        import pickle
        with open(mv.calibrator_tp_path, "rb") as f:
            cal_tp = pickle.load(f)
    if mv.calibrator_sl_path and Path(mv.calibrator_sl_path).exists():
        import pickle
        with open(mv.calibrator_sl_path, "rb") as f:
            cal_sl = pickle.load(f)

    return mv, tp, sl, cal_tp, cal_sl