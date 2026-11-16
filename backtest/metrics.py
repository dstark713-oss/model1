from __future__ import annotations

import numpy as np
import pandas as pd


def equity_curve(returns: pd.Series, initial: float = 1.0) -> pd.Series:
    return (1.0 + returns.fillna(0.0)).cumprod() * initial


def cagr(equity: pd.Series, periods_per_year: int = 252) -> float:
    if equity.empty or equity.iloc[0] <= 0:
        return np.nan
    n = len(equity)
    return float((equity.iloc[-1] / equity.iloc[0]) ** (periods_per_year / n) - 1.0)


def mdd(equity: pd.Series) -> float:
    if equity.empty:
        return np.nan
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def sharpe(returns: pd.Series, rf: float = 0.0, periods_per_year: int = 252) -> float:
    r = returns.dropna()
    if len(r) < 2 or r.std() == 0:
        return np.nan
    excess = r - rf / periods_per_year
    return float(excess.mean() / excess.std() * np.sqrt(periods_per_year))


def sortino(returns: pd.Series, periods_per_year: int = 252) -> float:
    r = returns.dropna()
    if len(r) < 2:
        return np.nan
    downside = r[r < 0].std()
    if downside == 0 or np.isnan(downside):
        return np.nan
    return float(r.mean() / downside * np.sqrt(periods_per_year))


def calmar(equity: pd.Series, periods_per_year: int = 252) -> float:
    m = mdd(equity)
    if m == 0 or np.isnan(m):
        return np.nan
    return float(cagr(equity, periods_per_year) / abs(m))


def summarize(df: pd.DataFrame, initial: float = 1.0) -> dict:
    """
    df: date-indexed, колонки ret_net, turnover, costs, exposure, cash.
    """
    ret = df["ret_net"].fillna(0.0)
    eq = equity_curve(ret, initial)
    return {
        "n_days": int(len(df)),
        "cagr": cagr(eq),
        "mdd": mdd(eq),
        "sharpe": sharpe(ret),
        "sortino": sortino(ret),
        "calmar": calmar(eq),
        "vol_ann": float(ret.std() * np.sqrt(252)),
        "total_return": float(eq.iloc[-1] / eq.iloc[0] - 1.0) if len(eq) else np.nan,
        "avg_turnover": float(df["turnover"].mean()),
        "total_costs": float(df["costs"].sum()),
        "avg_exposure": float(df["exposure"].mean()),
        "avg_cash": float(df["cash"].mean()),
    }