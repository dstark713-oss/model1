from __future__ import annotations

import json
import logging

from openai import OpenAI

log = logging.getLogger(__name__)


def _client(cfg: dict) -> OpenAI:
    key = cfg["openai"]["api_key"]
    if not key or key == "YOUR_OPENAI_API_KEY":
        raise RuntimeError(
            "OpenAI API key not set. Set OPENAI_API_KEY env or edit configs/agent.yaml."
        )
    kwargs = {"api_key": key}
    if cfg["openai"].get("base_url"):
        kwargs["base_url"] = cfg["openai"]["base_url"]
    return OpenAI(**kwargs)


def judge_ticker(ticker: str, cfg: dict) -> dict:
    """
    Модель через web_search ищет новости по тикеру и возвращает
    {"block": bool, "reason": str, "confidence": float, "sources": list[str]}.
    """
    client = _client(cfg)

    tools = []
    if cfg["web_search"]["enabled"]:
        tools.append({
            "type": "web_search",
            "search_context_size": cfg["web_search"]["search_context_size"],
        })

    system = cfg["prompt"]["system"].format(ticker=ticker)
    user = (
        f"Тикер: {ticker}. Найди свежие новости за последние 7 дней "
        f"и дай вердикт по блокировке. Ответ строго JSON."
    )

    resp = client.responses.create(
        model=cfg["openai"]["model"],
        temperature=cfg["openai"]["temperature"],
        max_output_tokens=cfg["openai"]["max_tokens"],
        timeout=cfg["openai"]["timeout_sec"],
        tools=tools,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )

    text = resp.output_text or "{}"
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        log.warning("[agent] %s: bad json: %s", ticker, text[:200])
        return {"block": False, "reason": "llm_parse_error",
                "confidence": 0.0, "sources": []}

    return {
        "block": bool(data.get("block", False)),
        "reason": str(data.get("reason", ""))[:300],
        "confidence": float(data.get("confidence", 0.0)),
        "sources": list(data.get("sources", []))[:10],
    }