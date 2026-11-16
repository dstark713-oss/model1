from __future__ import annotations

import json
import logging
from pathlib import Path

from agent.llm import judge_ticker

log = logging.getLogger(__name__)


def _latest_signal(signals_dir: Path) -> Path | None:
    files = sorted(signals_dir.glob("signal_*.json"))
    return files[-1] if files else None


def filter_signal(signal: dict, cfg: dict) -> dict:
    blocked = []
    kept = []

    for s in signal.get("selected", []):
        ticker = s["ticker"]
        try:
            verdict = judge_ticker(ticker, cfg)
        except RuntimeError as e:
            log.warning("[agent] LLM unavailable: %s", e)
            verdict = {"block": False, "reason": f"llm_unavailable: {e}",
                       "confidence": 0.0, "sources": []}
        except Exception as e:
            log.warning("[agent] %s: LLM error: %s", ticker, e)
            verdict = {"block": False, "reason": "llm_error",
                       "confidence": 0.0, "sources": []}

        if verdict["block"]:
            blocked.append({
                "ticker": ticker,
                "reason": verdict["reason"],
                "confidence": verdict["confidence"],
                "sources": verdict["sources"],
                "weight_freed": s["target_weight"],
            })
            log.info("[agent] BLOCK %s: %s (conf=%.2f)",
                     ticker, verdict["reason"], verdict["confidence"])
        else:
            kept.append(s)

    new_signal = dict(signal)
    new_signal["selected"] = kept
    new_signal["blocked"] = blocked
    new_signal["cash"] = float(signal.get("cash", 0.0)) + sum(b["weight_freed"] for b in blocked)
    new_signal["total_exposure"] = float(sum(s["target_weight"] for s in kept))
    return new_signal


def run_agent(signals_dir: Path, out_dir: Path, cfg: dict) -> dict | None:
    sig_path = _latest_signal(signals_dir)
    if sig_path is None:
        log.error("[agent] no signal file in %s", signals_dir)
        return None
    signal = json.loads(sig_path.read_text())
    log.info("[agent] signal %s, %d tickers", sig_path.name, len(signal.get("selected", [])))

    result = filter_signal(signal, cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"agent_{signal['date']}.json"
    p.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    log.info("[agent] saved -> %s", p)
    return result