# model1/llm_scripts/inspect_daily_state.py
"""Read-only inventory of local model1 artifacts and SQLite data for agent review.

Run from the model1 repository root:
    python -m llm_scripts.inspect_daily_state --date 2026-11-18

This script does not update market data, generate signals, or modify any files.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from urllib.parse import quote

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def section(title: str) -> None:
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def inspect_json_files(target_date: str, signal_date: str | None) -> None:
    section("SIGNAL / AGENT JSON")
    for folder in (ROOT / "artifacts" / "signals", ROOT / "artifacts" / "agent",
                   ROOT / "artifacts" / "live"):
        print(f"\n[{folder.relative_to(ROOT)}]")
        if not folder.exists():
            print("  MISSING")
            continue
        files = sorted(folder.glob("*.json"))
        if not files:
            print("  no JSON files")
            continue
        for path in files[-10:]:
            print(f"  {path.name} ({path.stat().st_size} bytes)")
        dates = list(dict.fromkeys([d for d in (signal_date, target_date) if d]))
        candidates = []
        for date in dates:
            candidates.extend([folder / f"signal_{date}.json",
                               folder / f"agent_{date}.json"])
        for path in candidates:
            if path.exists():
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    print(f"\n--- {path.name} ---")
                    print(json.dumps(data, ensure_ascii=False, indent=2))
                except Exception as exc:
                    print(f"  ERROR reading {path.name}: {exc}")


def inspect_parquet(target_date: str) -> None:
    section("PARQUET ARTIFACTS")
    paths = [
        ROOT / "artifacts" / "phase2" / "top100_pit.parquet",
        ROOT / "artifacts" / "phase7" / "backtest_daily.parquet",
        ROOT / "artifacts" / "phase5" / "oos_predictions.parquet",
    ]
    for path in paths:
        print(f"\n[{path.relative_to(ROOT)}]")
        if not path.exists():
            print("  MISSING")
            continue
        try:
            df = pd.read_parquet(path)
            print(f"  rows={len(df):,}; columns={list(df.columns)}")
            if "date" in df.columns:
                dates = pd.to_datetime(df["date"], errors="coerce")
                print(f"  date_range={dates.min()} .. {dates.max()}")
                day = df.loc[dates.dt.strftime("%Y-%m-%d") == target_date]
                print(f"  rows_on_target_date={len(day):,}")
                if not day.empty:
                    print(day.head(30).to_string(index=False))
            elif isinstance(df.index, pd.DatetimeIndex):
                print(f"  index_range={df.index.min()} .. {df.index.max()}")
                if target_date in df.index.strftime("%Y-%m-%d"):
                    print(df.loc[df.index.strftime("%Y-%m-%d") == target_date].head(30).to_string())
        except Exception as exc:
            print(f"  ERROR reading parquet: {type(exc).__name__}: {exc}")


def inspect_sqlite(target_date: str) -> None:
    section("SQLITE DATABASE")
    db = ROOT / "data" / "tactical.sqlite"
    print(f"DB: {db}")
    if not db.exists():
        print("  MISSING")
        return

    uri = "file:" + quote(str(db.resolve()).replace("\\", "/")) + "?mode=ro"
    try:
        con = sqlite3.connect(uri, uri=True)
        con.row_factory = sqlite3.Row
        tables = con.execute(
            "SELECT name, sql FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        for table in tables:
            name = table["name"]
            print(f"\n[TABLE {name}]")
            print(table["sql"])
            try:
                count = con.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]
                print(f"rows={count:,}")
            except Exception as exc:
                print(f"count_error={exc}")
            cols = con.execute(f'PRAGMA table_info("{name}")').fetchall()
            col_names = [r["name"] for r in cols]
            print(f"columns={col_names}")

            date_col = next((c for c in ("date", "trade_date", "tradedate", "TRADEDATE")
                             if c in col_names), None)
            if date_col:
                try:
                    bounds = con.execute(
                        f'SELECT MIN("{date_col}"), MAX("{date_col}") FROM "{name}"'
                    ).fetchone()
                    print(f"date_range={tuple(bounds)}")
                except Exception as exc:
                    print(f"date_range_error={exc}")

            if name.lower() == "prices":
                try:
                    sample = con.execute(
                        f'SELECT * FROM "{name}" ORDER BY "{date_col or col_names[0]}" DESC LIMIT 8'
                    ).fetchall()
                    for row in sample:
                        print(dict(row))
                except Exception as exc:
                    print(f"sample_error={exc}")
        con.close()
    except Exception as exc:
        print(f"  ERROR opening SQLite read-only: {type(exc).__name__}: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="2026-11-18",
                        help="portfolio performance date")
    parser.add_argument("--signal-date", default="2026-11-17",
                        help="signal intended for the performance date")
    args = parser.parse_args()
    print(f"MODEL1 DAILY STATE INSPECTION | target_date={args.date} | "
          f"signal_date={args.signal_date} | READ ONLY")
    inspect_json_files(args.date, args.signal_date)
    inspect_parquet(args.date)
    inspect_sqlite(args.date)


if __name__ == "__main__":
    main()
