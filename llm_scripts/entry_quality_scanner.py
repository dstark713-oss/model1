# model1/llm_scripts/entry_quality_scanner.py
"""Descriptive entry-quality scanner for the discretionary AI Quant Lab.

Run from the model1 repository root:
    python llm_scripts/entry_quality_scanner.py
    python llm_scripts/entry_quality_scanner.py --date 2026-11-18
    python llm_scripts/entry_quality_scanner.py --tickers T,X5,MDMG,GCHE

READ-ONLY with respect to SQLite and model artifacts.
Writes a dated JSON report to artifacts/agent/.
Does not generate ML signals, predict returns, recommend weights, or place trades.

The score is a transparent heuristic summary of current technical conditions.
It is NOT a probability, expected return, or validated trading edge.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median, stdev

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_TICKERS = {"IMOEX", "RGBI"}


def finite_float(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def pct_change(old, new):
    if old is None or new is None or old <= 0:
        return None
    return new / old - 1.0


def safe_ratio(a, b):
    if a is None or b is None or b == 0:
        return None
    return a / b


def load_bars(con, as_of):
    rows = con.execute(
        """
        SELECT ticker, date, open, high, low, close, volume, value
        FROM prices
        WHERE date <= ?
          AND close IS NOT NULL
          AND close > 0
        ORDER BY ticker, date
        """,
        (as_of,),
    ).fetchall()

    result = {}
    for row in rows:
        ticker, date, opn, high, low, close, volume, value = row
        result.setdefault(ticker, []).append({
            "date": date,
            "open": finite_float(opn),
            "high": finite_float(high),
            "low": finite_float(low),
            "close": finite_float(close),
            "volume": finite_float(volume),
            "value": finite_float(value),
        })
    return result


def average_turnover(bars, window=20):
    values = []
    for bar in bars[-window:]:
        value = bar["value"]
        if value is None and bar["volume"] is not None:
            value = bar["close"] * bar["volume"]
        if value is not None and value > 0:
            values.append(value)
    return mean(values) if values else None


def calculate_features(bars):
    bars = [b for b in bars if b["close"] is not None and b["close"] > 0]
    closes = [b["close"] for b in bars]
    if len(closes) < 60:
        return None

    daily_returns = [
        closes[i] / closes[i - 1] - 1.0
        for i in range(1, len(closes))
        if closes[i - 1] > 0
    ]
    if len(daily_returns) < 50:
        return None

    sma20_series = [
        mean(closes[i - 19:i + 1])
        for i in range(19, len(closes))
    ]
    sma20 = sma20_series[-1]
    sma20_5d_ago = sma20_series[-6] if len(sma20_series) >= 6 else None
    sma50 = mean(closes[-50:])

    last = bars[-1]
    high20_values = [
        b["high"] if b["high"] is not None else b["close"]
        for b in bars[-20:]
    ]
    high20 = max(high20_values)

    volumes = [b["volume"] for b in bars[-21:-1] if b["volume"] is not None and b["volume"] > 0]
    volume_median20 = median(volumes) if volumes else None
    volume_ratio = safe_ratio(last["volume"], volume_median20)

    candle_location = None
    if last["high"] is not None and last["low"] is not None:
        candle_range = last["high"] - last["low"]
        if candle_range > 0:
            candle_location = (last["close"] - last["low"]) / candle_range

    return {
        "last_date": last["date"],
        "close": closes[-1],
        "ret_5": pct_change(closes[-6], closes[-1]),
        "ret_20": pct_change(closes[-21], closes[-1]),
        "ret_60": pct_change(closes[-61], closes[-1]) if len(closes) >= 61 else None,
        "sma20": sma20,
        "sma50": sma50,
        "sma20_slope_5": pct_change(sma20_5d_ago, sma20) if sma20_5d_ago else None,
        "distance_sma20": pct_change(sma20, closes[-1]),
        "distance_sma50": pct_change(sma50, closes[-1]),
        "distance_high20": pct_change(high20, closes[-1]),
        "volatility_20_ann": stdev(daily_returns[-20:]) * math.sqrt(252),
        "avg_value_20": average_turnover(bars, 20),
        "volume_ratio_vs_median20": volume_ratio,
        "last_candle_close_location": candle_location,
        "observations": len(closes),
    }


def score_features(f, benchmark):
    components = {}
    reasons = []
    warnings = []

    # Trend: 25 points.
    trend = 0
    if f["close"] > f["sma20"]:
        trend += 8
        reasons.append("цена выше SMA20")
    else:
        warnings.append("цена ниже SMA20")
    if f["sma20"] > f["sma50"]:
        trend += 9
        reasons.append("SMA20 выше SMA50")
    else:
        warnings.append("SMA20 не выше SMA50")
    if f["sma20_slope_5"] is not None and f["sma20_slope_5"] > 0:
        trend += 8
        reasons.append("SMA20 растёт")
    else:
        warnings.append("нет положительного наклона SMA20")
    components["trend_25"] = trend

    # Momentum: 20 points.
    momentum = 0
    for key, points, label in (
        ("ret_5", 4, "положительная доходность за 5 сессий"),
        ("ret_20", 8, "положительная доходность за 20 сессий"),
        ("ret_60", 8, "положительная доходность за 60 сессий"),
    ):
        if f[key] is not None and f[key] > 0:
            momentum += points
            reasons.append(label)
        elif f[key] is not None:
            warnings.append(f"отрицательная динамика {key}")
    components["momentum_20"] = momentum

    # Relative strength versus IMOEX: 15 points, omitted if benchmark unavailable.
    if benchmark is not None:
        relative = 0
        if f["ret_20"] is not None and benchmark["ret_20"] is not None:
            if f["ret_20"] > benchmark["ret_20"]:
                relative += 8
                reasons.append("опережает IMOEX за 20 сессий")
            else:
                warnings.append("уступает IMOEX за 20 сессий")
        if f["ret_60"] is not None and benchmark["ret_60"] is not None:
            if f["ret_60"] > benchmark["ret_60"]:
                relative += 7
                reasons.append("опережает IMOEX за 60 сессий")
            else:
                warnings.append("уступает IMOEX за 60 сессий")
        components["relative_strength_15"] = relative

    # Volume confirmation: 15 points.
    volume_score = 0
    ratio = f["volume_ratio_vs_median20"]
    if ratio is not None:
        if f["ret_5"] is not None and f["ret_5"] > 0 and ratio >= 1.2:
            volume_score += 10
            reasons.append("рост сопровождается повышенным объёмом")
        elif ratio >= 0.8:
            volume_score += 5
            reasons.append("объём близок к обычному уровню")
        else:
            warnings.append("объём ниже медианного")
    else:
        warnings.append("недостаточно данных об объёме")
    location = f["last_candle_close_location"]
    if location is not None:
        if location >= 0.7:
            volume_score += 5
            reasons.append("закрытие последней свечи в верхней части диапазона")
        elif location < 0.35:
            warnings.append("закрытие последней свечи в нижней части диапазона")
    components["volume_confirmation_15"] = volume_score

    # Entry location: 15 points. Avoids rewarding a very extended move.
    distance = f["distance_high20"]
    location_score = 0
    if distance is not None:
        if distance >= -0.03:
            location_score = 10
            warnings.append("цена близка к 20-дневному максимуму: риск покупки на выносе")
        elif distance >= -0.08:
            location_score = 15
            reasons.append("умеренный откат от 20-дневного максимума")
        elif distance >= -0.15:
            location_score = 8
            reasons.append("заметный откат от 20-дневного максимума")
        else:
            location_score = 2
            warnings.append("глубокая просадка от 20-дневного максимума")
    components["entry_location_15"] = location_score

    # Realized volatility: 10 points. This is a risk descriptor, not a forecast.
    vol = f["volatility_20_ann"]
    if vol <= 0.20:
        risk_score = 10
    elif vol <= 0.35:
        risk_score = 8
    elif vol <= 0.50:
        risk_score = 5
    elif vol <= 0.70:
        risk_score = 2
    else:
        risk_score = 0
        warnings.append("очень высокая реализованная волатильность")
    components["risk_10"] = risk_score

    available = sum(
        {"trend_25": 25, "momentum_20": 20, "relative_strength_15": 15,
         "volume_confirmation_15": 15, "entry_location_15": 15, "risk_10": 10}[k]
        for k in components
    )
    earned = sum(components.values())
    score = 100.0 * earned / available if available else None

    if score is None:
        label = "UNRATED"
    elif score >= 75:
        label = "STRONG"
    elif score >= 55:
        label = "MODERATE"
    elif score >= 35:
        label = "WEAK"
    else:
        label = "POOR"

    return {
        "score": round(score, 1) if score is not None else None,
        "score_label": label,
        "points_earned": earned,
        "points_available": available,
        "components": components,
        "reasons": reasons,
        "warnings": warnings,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Descriptive technical entry-quality scan; not a trading signal."
    )
    parser.add_argument("--date", help="analysis date; defaults to latest date in prices")
    parser.add_argument("--db", default=str(ROOT / "data" / "tactical.sqlite"))
    parser.add_argument("--output-dir", default=str(ROOT / "artifacts" / "agent"))
    parser.add_argument("--min-avg-value", type=float, default=5_000_000,
                        help="minimum proxy average daily turnover in RUB")
    parser.add_argument("--tickers", help="optional comma-separated ticker allowlist")
    parser.add_argument("--top", type=int, default=30, help="number of rows to print")
    args = parser.parse_args()

    db = Path(args.db)
    if not db.exists():
        raise SystemExit(f"Database not found: {db}")

    con = sqlite3.connect(f"file:{db.resolve().as_posix()}?mode=ro", uri=True)
    latest_date = con.execute("SELECT MAX(date) FROM prices").fetchone()[0]
    as_of = args.date or latest_date
    bars_by_ticker = load_bars(con, as_of)
    con.close()

    if not bars_by_ticker:
        raise SystemExit(f"No price bars found on or before {as_of}")

    benchmark_bars = bars_by_ticker.get("IMOEX", [])
    benchmark = calculate_features(benchmark_bars) if benchmark_bars else None

    allowlist = None
    if args.tickers:
        allowlist = {x.strip().upper() for x in args.tickers.split(",") if x.strip()}

    results = []
    for ticker, bars in bars_by_ticker.items():
        if ticker in EXCLUDED_TICKERS:
            continue
        if allowlist is not None and ticker not in allowlist:
            continue

        f = calculate_features(bars)
        if f is None:
            continue
        if f["last_date"] != as_of:
            continue

        scored = score_features(f, benchmark)
        liquidity_ok = (
            f["avg_value_20"] is not None
            and f["avg_value_20"] >= args.min_avg_value
        )
        if not liquidity_ok:
            scored["warnings"].append("ниже минимального порога оборота")
        if f["ret_20"] is not None and f["ret_20"] >= 0.30:
            scored["warnings"].append("сильное движение за 20 сессий: не преследовать цену")
        if f["ret_5"] is not None and f["ret_5"] >= 0.15:
            scored["warnings"].append("резкий рост за 5 сессий: повышенный риск погони за ценой")

        results.append({
            "ticker": ticker,
            **f,
            **scored,
            "liquidity_ok": liquidity_ok,
            "eligible_for_review": liquidity_ok and scored["score"] is not None,
        })

    results.sort(
        key=lambda x: (
            x["eligible_for_review"],
            x["score"] if x["score"] is not None else -1,
            x["avg_value_20"] if x["avg_value_20"] is not None else -1,
        ),
        reverse=True,
    )

    report = {
        "as_of": as_of,
        "latest_db_date": latest_date,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "read_only_inputs": True,
        "method": {
            "name": "transparent_entry_quality_heuristic_v1",
            "score_range": "0-100",
            "not_a_probability": True,
            "not_expected_return": True,
            "not_a_buy_signal": True,
            "components_max_points": {
                "trend": 25, "momentum": 20, "relative_strength_vs_imoex": 15,
                "volume_confirmation": 15, "entry_location": 15, "realized_volatility_risk": 10,
            },
            "min_avg_value_proxy_rub": args.min_avg_value,
            "universe_note": "Uses tickers present in local prices table; instrument type is not independently filtered. Use --tickers to restrict the scan.",
        },
        "benchmark": benchmark,
        "candidate_count": len(results),
        "candidates": results,
        "limitations": [
            "Score is a heuristic ranking aid and has not been calibrated or validated against forward returns.",
            "No order book, bid-ask spread, slippage, intraday path, corporate-action adjustment, or broker fills.",
            "Turnover is close*volume when prices.value is NULL; it is not executable liquidity.",
            "Universe inherits local prices table and may include preferred shares, ETFs, or other instruments.",
            "No fundamentals, news, event risk, ML probabilities, or portfolio constraints are included.",
            "Features use observed bars; missing sessions are not forward-filled.",
        ],
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"entry_quality_{as_of}.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"ENTRY QUALITY SCANNER | as_of={as_of} | latest_db_date={latest_date}")
    print("Heuristic score only: NOT probability / expected return / buy signal.")
    print("ticker score label trend momentum relative volume location risk ret5 ret20 ret60 vol20 avg_value20")
    for item in results[:max(args.top, 0)]:
        c = item["components"]
        print(
            f"{item['ticker']:>8} {item['score'] if item['score'] is not None else 'n/a':>5} "
            f"{item['score_label']:>9} "
            f"{c.get('trend_25', 0):>2} {c.get('momentum_20', 0):>2} "
            f"{c.get('relative_strength_15', 'n/a')!s:>8} "
            f"{c.get('volume_confirmation_15', 0):>2} "
            f"{c.get('entry_location_15', 0):>3} {c.get('risk_10', 0):>2} "
            f"{item['ret_5']:+.2%} {item['ret_20']:+.2%} "
            f"{item['ret_60'] if item['ret_60'] is not None else float('nan'):+.2%} "
            f"{item['volatility_20_ann']:.2%} "
            f"{item['avg_value_20'] if item['avg_value_20'] is not None else float('nan'):.0f}"
        )
        if item["warnings"]:
            print("         WARN:", "; ".join(item["warnings"]))
    print(f"\nJSON report: {out_path}")
    print("No model signal was generated or changed. No trades were placed.")


if __name__ == "__main__":
    main()
