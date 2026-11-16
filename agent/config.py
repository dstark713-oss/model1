from __future__ import annotations

import os
from pathlib import Path

import yaml


def load_agent_config(path: str | Path = "configs/agent.yaml") -> dict:
    p = Path(path)
    if not p.is_absolute():
        p = Path(__file__).resolve().parent.parent / p
    cfg = yaml.safe_load(p.read_text())["agent"]
    # ключ: env перекрывает yaml
    env_key = os.environ.get("OPENAI_API_KEY")
    if env_key:
        cfg["openai"]["api_key"] = env_key
    return cfg