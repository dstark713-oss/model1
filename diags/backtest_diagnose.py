# diags/backtest_diagnose.py
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BT_PATH = PROJECT_ROOT / "artifacts" / "phase7" / "backtest_daily.parquet"
PREDS_PATH = PROJECT_ROOT / "artifacts" / "phase5" / "oos_predictions.parquet"


def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def diagnose_scores(preds: pd.DataFrame) -> None:
    section("1. SCORE / PREDICTIONS")
    preds = preds.copy()
    preds["score"] = preds["p_tp"] * (1 - preds["p_sl"])
    daily = preds.groupby("date")["score"].agg(["std", "min", "max", "count"])
    print(f"  score std   : mean={daily['std'].mean():.4f}  "
          f"p10={daily['std'].quantile(0.1):.4f}  p90={daily['std'].quantile(0.9):.4f}")
    print(f"  score range : mean(max-min)={((daily['max']-daily['min']).mean()):.4f}")
    print(f"  candidates/day: mean={daily['count'].mean():.1f}")
    print(f"  p_tp: mean={preds['p_tp'].mean():.4f} std={preds['p_tp'].std():.4f}")
    print(f"  p_sl: mean={preds['p_sl'].mean():.4f} std={preds['p_sl'].std():.4f}")


def diagnose_turnover(bt: pd.DataFrame) -> None:
    section("2. TURNOVER DECOMPOSITION")
    dates = bt.index.tolist()
    set_changes, exposure_changes = [], []
    for i in range(1, len(dates)):
        prev = pd.Series(bt.loc[dates[i-1], "target_w"])
        cur = pd.Series(bt.loc[dates[i], "target_w"])
        prev_set = set(prev[prev > 0].index)
        cur_set = set(cur[cur > 0].index)
        set_changes.append(len(prev_set.symmetric_difference(cur_set)))
        exposure_changes.append(abs(bt.loc[dates[i], "exposure"] - bt.loc[dates[i-1], "exposure"]))
    print(f"  avg tickers changed/day: {np.mean(set_changes):.2f}")
    print(f"  avg |Δexposure|/day   : {np.mean(exposure_changes):.4f}")
    print(f"  avg turnover/day       : {bt['turnover'].mean():.4f}")
    print(f"  turnover p50/p90/p99   : "
          f"{bt['turnover'].quantile(0.5):.3f}/"
          f"{bt['turnover'].quantile(0.9):.3f}/"
          f"{bt['turnover'].quantile(0.99):.3f}")


def diagnose_costs(bt: pd.DataFrame) -> None:
    section("3. COSTS vs GROSS")
    gross = bt["ret_gross"].fillna(0).sum()
    costs = bt["costs"].fillna(0).sum()
    net = bt["ret_net"].fillna(0).sum()
    print(f"  sum gross return : {gross:.4f}")
    print(f"  sum costs        : {costs:.4f}")
    print(f"  sum net return   : {net:.4f}")
    if gross != 0:
        print(f"  costs/gross      : {costs / gross * 100:.1f}%")


def diagnose_yearly(bt: pd.DataFrame) -> None:
    section("4. EQUITY BY YEAR")
    ret = bt["ret_net"].fillna(0)
    eq = (1 + ret).cumprod()
    yearly = eq.resample("YE").last()
    yearly_ret = yearly.pct_change()
    if len(yearly) > 0:
        yearly_ret.iloc[0] = yearly.iloc[0] - 1.0
    for dt, r in yearly_ret.items():
        bar = "█" * int(abs(r) * 50)
        sign = "+" if r >= 0 else "-"
        print(f"  {dt.year}: {r*100:+7.2f}%  {sign}{bar}")


def diagnose_drawdown(bt: pd.DataFrame) -> None:
    section("5. DRAWDOWN DETAIL")
    ret = bt["ret_net"].fillna(0)
    eq = (1 + ret).cumprod()
    peak = eq.cummax()
    dd = eq / peak - 1.0
    print(f"  max dd     : {dd.min():.4f} on {dd.idxmin().date()}")
    print(f"  dd p95     : {dd.quantile(0.05):.4f}")
    under_water = (dd < -0.05).sum()
    print(f"  days dd<-5%: {under_water} ({under_water/len(dd)*100:.1f}%)")

    print("  top-5 drawdown troughs:")
    dd_neg = dd[dd < 0].copy()
    if dd_neg.empty:
        return
    # группируем последовательные периоды под водой
    groups = (dd_neg.diff() > 0).cumsum()
    troughs = dd_neg.groupby(groups).apply(
        lambda s: pd.Series({"trough_date": s.idxmin(), "depth": s.min()}),
        include_groups=False,
    )
    for _, row in troughs.nsmallest(5, "depth").iterrows():
        print(f"    {row['trough_date'].date()}: {row['depth']*100:.2f}%")


def main(argv=None) -> int:
    if not BT_PATH.exists():
        print(f"[diagnose] missing {BT_PATH}")
        return 1
    bt = pd.read_parquet(BT_PATH)
    if bt.empty:
        print("[diagnose] empty backtest")
        return 1
    preds = pd.read_parquet(PREDS_PATH) if PREDS_PATH.exists() else pd.DataFrame()

    if not preds.empty:
        diagnose_scores(preds)
    diagnose_turnover(bt)
    diagnose_costs(bt)
    diagnose_yearly(bt)
    diagnose_drawdown(bt)
    return 0


if __name__ == "__main__":
    sys.exit(main())