from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from agent.config import load_agent_config
from agent.filter import run_agent

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    p = argparse.ArgumentParser(prog="agent")
    p.add_argument("--config", default="configs/agent.yaml")
    p.add_argument("--signals-dir", default="artifacts/signals")
    p.add_argument("--out-dir", default=None)
    args = p.parse_args(argv)

    cfg = load_agent_config(args.config)
    if not cfg.get("enabled", True):
        print("[agent] disabled in config")
        return 0

    signals_dir = Path(args.signals_dir)
    if not signals_dir.is_absolute():
        signals_dir = PROJECT_ROOT / signals_dir
    out_dir = Path(args.out_dir) if args.out_dir else PROJECT_ROOT / cfg["output_dir"]

    result = run_agent(signals_dir, out_dir, cfg)
    if result is None:
        return 1

    print("\n" + "=" * 60)
    print(f"AGENT  {result['date']}")
    print("=" * 60)
    if result["blocked"]:
        for b in result["blocked"]:
            print(f"  BLOCK {b['ticker']:10s} freed={b['weight_freed']:.4f} "
                  f"conf={b['confidence']:.2f}  {b['reason']}")
    else:
        print("  (nothing blocked)")
    print(f"  kept: {len(result['selected'])}  cash: {result['cash']:.4f}  "
          f"exposure: {result['total_exposure']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())