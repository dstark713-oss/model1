# model1/llm_scripts/daily_agent_review.py
"""Daily decision-support packet for the discretionary AI Quant Lab portfolio.

Run from the model1 repository root:
    python llm_scripts/daily_agent_review.py
    python llm_scripts/daily_agent_review.py --date 2026-11-18

READ-ONLY with respect to the market database and model artifacts.
Writes only a dated JSON review packet to artifacts/agent/.
It does NOT generate or modify a model signal and does NOT place trades.

The scanner is a research aid, not an allocation engine:
- ML signal is treated as one input, not as the portfolio decision.
- Technical features are computed from observed bars (not calendar days).
- Liquidity is proxied by close * volume because prices.value may be NULL.
- The securities table may contain non-common shares / non-equity instruments;
  candidate lists require instrument and fundamental review before use.
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = {"T": 0.10, "X5": 0.10, "ASTR": 0.05, "GCHE": 0.025}


def pct_change(a: float | None, b: float | None) -> float | None:
    if a is None or b is None or a == 0:
        return None
    return b / a - 1.0


def safe_float(value):
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def load_signal(folder: Path):
    files = sorted(folder.glob("signal_*.json")) if folder.exists() else []
    parsed = []
    for path in files:
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
            parsed.append((str(obj.get("date", "")), path, obj))
        except Exception as exc:
            print(f"WARNING: cannot parse {path.name}: {exc}")
    if not parsed:
        return None, None
    parsed.sort(key=lambda x: (x[0], x[1].name))
    _, path, obj = parsed[-1]
    return path, obj


def load_prices(con: sqlite3.Connection):
    rows = con.execute("""
        SELECT ticker, date, open, high, low, close, volume, value
        FROM prices
        WHERE close IS NOT NULL AND close > 0
        ORDER BY ticker, date
    """).fetchall()
    by_ticker = {}
    for ticker, date, opn, high, low, close, volume, value in rows:
        by_ticker.setdefault(ticker, []).append({
            "date": date, "open": safe_float(opn), "high": safe_float(high),
            "low": safe_float(low), "close": safe_float(close),
            "volume": safe_float(volume), "value": safe_float(value),
        })
    return by_ticker


def returns_and_features(bars):
    closes = [b["close"] for b in bars if b["close"] is not None]
    if len(closes) < 22:
        return None
    daily = [closes[i] / closes[i - 1] - 1 for i in range(1, len(closes))
             if closes[i - 1] and closes[i] and closes[i - 1] > 0]
    if len(daily) < 21:
        return None
    last = bars[-1]
    values = []
    for b in bars[-20:]:
        v = b["value"]
        if v is None and b["volume"] is not None and b["close"] is not None:
            v = b["volume"] * b["close"]
        if v is not None and v > 0:
            values.append(v)
    vol20 = stdev(daily[-20:]) * math.sqrt(252) if len(daily[-20:]) > 1 else None
    return {
        "last_date": last["date"],
        "close": closes[-1],
        "ret_5": pct_change(closes[-6], closes[-1]) if len(closes) >= 6 else None,
        "ret_20": pct_change(closes[-21], closes[-1]) if len(closes) >= 21 else None,
        "ret_60": pct_change(closes[-61], closes[-1]) if len(closes) >= 61 else None,
        "volatility_20_ann": vol20,
        "avg_value_20": mean(values) if values else None,
        "observations": len(closes),
    }


def aligned_corr(bars_a, bars_b, window=60):
    a = {b["date"]: b["close"] for b in bars_a if b["close"]}
    b = {x["date"]: x["close"] for x in bars_b if x["close"]}
    dates = sorted(set(a) & set(b))[-(window + 1):]
    if len(dates) < 15:
        return None
    ra = [a[dates[i]] / a[dates[i - 1]] - 1 for i in range(1, len(dates))]
    rb = [b[dates[i]] / b[dates[i - 1]] - 1 for i in range(1, len(dates))]
    ma, mb = mean(ra), mean(rb)
    da = sum((x - ma) ** 2 for x in ra)
    db = sum((x - mb) ** 2 for x in rb)
    if da <= 0 or db <= 0:
        return None
    return sum((x - ma) * (y - mb) for x, y in zip(ra, rb)) / math.sqrt(da * db)


def fmt(x, digits=2):
    return "n/a" if x is None else f"{x * 100:.{digits}f}%"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", help="review date; defaults to latest date in prices")
    parser.add_argument("--db", default=str(ROOT / "data" / "tactical.sqlite"))
    parser.add_argument("--signal-dir", default=str(ROOT / "artifacts" / "signals"))
    parser.add_argument("--output-dir", default=str(ROOT / "artifacts" / "agent"))
    parser.add_argument("--min-avg-value", type=float, default=5_000_000,
                        help="minimum proxy average daily traded value, RUB")
    args = parser.parse_args()

    db = Path(args.db)
    if not db.exists():
        raise SystemExit(f"Database not found: {db}")
    con = sqlite3.connect(f"file:{db.resolve().as_posix()}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    latest_db_date = con.execute("SELECT MAX(date) FROM prices").fetchone()[0]
    review_date = args.date or latest_db_date
    by_ticker = load_prices(con)
    securities = {
        r["ticker"]: {"description": r["description"], "source": r["source"]}
        for r in con.execute("SELECT ticker, source, description FROM securities")
    }
    con.close()

    features = {}
    for ticker, bars in by_ticker.items():
        # Use only bars up to the requested review date.
        eligible = [b for b in bars if b["date"] <= review_date]
        f = returns_and_features(eligible)
        if f:
            features[ticker] = f

    if not features:
        raise SystemExit(f"No usable price history on or before {review_date}")

    market_date = max(f["last_date"] for f in features.values())
    signal_path, signal = load_signal(Path(args.signal_dir))
    signal_date = signal.get("date") if signal else None
    signal_map = {}
    if signal:
        for item in signal.get("score_ranking", []):
            signal_map[item.get("ticker")] = item

    imoex = features.get("IMOEX")
    portfolio = []
    portfolio_return = 0.0
    missing_positions = []
    for ticker, weight in DEFAULT_WEIGHTS.items():
        f = features.get(ticker)
        bars = [b for b in by_ticker.get(ticker, []) if b["date"] <= review_date]
        # Return from previous observed close to review-date close.
        ret = None
        if len(bars) >= 2 and bars[-1]["date"] == review_date:
            ret = pct_change(bars[-2]["close"], bars[-1]["close"])
        if ret is None:
            missing_positions.append(ticker)
        else:
            portfolio_return += weight * ret
        portfolio.append({
            "ticker": ticker, "target_weight": weight,
            "daily_return": ret,
            "contribution": weight * ret if ret is not None else None,
            "features": f,
            "ml": signal_map.get(ticker),
            "description": securities.get(ticker, {}).get("description"),
        })

    candidates = []
    for ticker, f in features.items():
        if ticker in DEFAULT_WEIGHTS or ticker in ("IMOEX", "RGBI"):
            continue
        if f["last_date"] != review_date:
            continue
        if f["observations"] < 60 or f["avg_value_20"] is None:
            continue
        if f["avg_value_20"] < args.min_avg_value:
            continue
        # Transparent descriptive ranking: momentum adjusted by realized vol.
        # Not a probability, not an expected return, not a buy signal.
        score = None
        if f["ret_20"] is not None and f["volatility_20_ann"] and f["volatility_20_ann"] > 0:
            score = f["ret_20"] / f["volatility_20_ann"]
        candidates.append({
            "ticker": ticker, **f, "momentum_vol_ratio_20": score,
            "ml": signal_map.get(ticker),
            "description": securities.get(ticker, {}).get("description"),
        })
    candidates.sort(key=lambda x: (
        x["momentum_vol_ratio_20"] is not None,
        x["momentum_vol_ratio_20"] if x["momentum_vol_ratio_20"] is not None else -999
    ), reverse=True)

    held_corr = []
    held = list(DEFAULT_WEIGHTS)
    for i, a in enumerate(held):
        for b in held[i + 1:]:
            corr = aligned_corr(
                [x for x in by_ticker.get(a, []) if x["date"] <= review_date],
                [x for x in by_ticker.get(b, []) if x["date"] <= review_date],
            )
            held_corr.append({"a": a, "b": b, "corr_60": corr})

    signal_summary = None
    if signal:
        signal_summary = {
            "file": signal_path.name,
            "date": signal_date,
            "regime": signal.get("regime"),
            "regime_date": signal.get("regime_date"),
            "market_breadth": signal.get("market_breadth"),
            "n_star": signal.get("n_star"),
            "model_version": signal.get("model_version"),
            "selected": signal.get("selected", signal.get("buy_list", [])),
            "top_ranking": signal.get("score_ranking", [])[:20],
        }

    market_breadth_20 = None
    comparable = [f for t, f in features.items()
                  if f["last_date"] == review_date and f["ret_20"] is not None
                  and t not in ("IMOEX", "RGBI")]
    if comparable:
        market_breadth_20 = sum(1 for f in comparable if f["ret_20"] > 0) / len(comparable)

    report = {
        "review_date": review_date,
        "latest_db_date": latest_db_date,
        "latest_observed_price_date": market_date,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "read_only_inputs": True,
        "portfolio_weights": {**DEFAULT_WEIGHTS, "CASH": 1 - sum(DEFAULT_WEIGHTS.values())},
        "portfolio_close_to_close_return": portfolio_return if not missing_positions else None,
        "portfolio_positions": portfolio,
        "missing_position_prices_on_review_date": missing_positions,
        "imoex": imoex,
        "market_breadth_20d_positive": market_breadth_20,
        "held_pairwise_correlations_60": held_corr,
        "ml_signal": signal_summary,
        "technical_candidate_scan": {
            "description": "20-session momentum / 20-session annualized realized volatility; descriptive ranking only",
            "min_avg_value_proxy_rub": args.min_avg_value,
            "candidate_count": len(candidates),
            "top_30": candidates[:30],
        },
        "limitations": [
            "No order book, spreads, slippage, intraday path, corporate-action adjustment, or broker fills.",
            "Liquidity proxy is close*volume when prices.value is NULL.",
            "Universe inherits the local securities/prices database and may contain non-common shares or non-equity instruments.",
            "Technical ranking is not an ML prediction or buy signal.",
            "Fundamental/news/event checks are not automated in this packet and must be performed before final allocation.",
            "Portfolio return assumes the configured target weights were held from prior close to review close; cash return is zero."
        ],
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"daily_review_{review_date}.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"DAILY AGENT REVIEW | review_date={review_date} | price_db_latest={latest_db_date}")
    print(f"ML signal: {signal_date or 'MISSING'} | file={signal_path.name if signal_path else 'MISSING'}")
    if signal:
        print(f"Regime={signal.get('regime')} | breadth={signal.get('market_breadth')} | n*={signal.get('n_star')}")
        print("ML ranking: rank ticker score p_tp p_sl selected")
        for item in signal.get("score_ranking", [])[:20]:
            print(f"{item.get('rank', item.get('score_rank')):>4} {item.get('ticker',''):>8} "
                  f"{item.get('score', 0):>9.5f} {item.get('p_tp', 0):>7.4f} "
                  f"{item.get('p_sl', 0):>7.4f} {item.get('selected', False)}")
    print("\nDISCRETIONARY POSITIONS | prior-close to review-close")
    print("ticker weight return contribution ret20 ret60 vol20 avg_value20")
    for p in portfolio:
        f = p["features"] or {}
        print(f"{p['ticker']:>6} {p['target_weight']:>6.1%} {fmt(p['daily_return']):>9} "
              f"{fmt(p['contribution']):>11} {fmt(f.get('ret_20')):>8} "
              f"{fmt(f.get('ret_60')):>8} {fmt(f.get('volatility_20_ann')):>8} "
              f"{f.get('avg_value_20') if f.get('avg_value_20') is not None else 'n/a'}")
    print(f"Calculated portfolio return: {fmt(report['portfolio_close_to_close_return'])}")
    print(f"IMOEX close: {imoex.get('close') if imoex else 'n/a'} | IMOEX ret20: {fmt(imoex.get('ret_20') if imoex else None)}")
    print(f"Market breadth 20d positive: {fmt(market_breadth_20)}")
    print("\nHELD PAIRWISE CORRELATIONS (60 observations)")
    for x in held_corr:
        print(f"{x['a']:>5} / {x['b']:<5}: {x['corr_60'] if x['corr_60'] is not None else 'n/a'}")
    print("\nTECHNICAL CANDIDATES (NOT BUY SIGNALS)")
    print("ticker ret5 ret20 ret60 vol20 avg_value20 mom/vol ML-rank")
    for c in candidates[:20]:
        ml_rank = c["ml"].get("rank") if c.get("ml") else None
        print(f"{c['ticker']:>8} {fmt(c['ret_5']):>8} {fmt(c['ret_20']):>8} "
              f"{fmt(c['ret_60']):>8} {fmt(c['volatility_20_ann']):>8} "
              f"{c['avg_value_20']:>14.0f} "
              f"{c['momentum_vol_ratio_20'] if c['momentum_vol_ratio_20'] is not None else 'n/a':>9} "
              f"{ml_rank if ml_rank is not None else '-'}")
    print(f"\nJSON review packet: {out_path}")
    print("No model signal was generated or changed. No trades were placed.")


if __name__ == "__main__":
    main()
