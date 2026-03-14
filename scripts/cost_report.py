#!/usr/bin/env python3
"""Generate cost report from cost_log.jsonl.
Usage: poetry run python scripts/cost_report.py [--log cost_log.jsonl] [--phase phase_N]
"""
from __future__ import annotations
import json
import sys
from collections import defaultdict
from pathlib import Path
import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Cost report from cost_log.jsonl")
    parser.add_argument("--log", default="cost_log.jsonl")
    parser.add_argument("--phase", default=None, help="Filter by phase name")
    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        print(f"No cost log found at {log_path}. Run make explain TX=<id> first.")
        sys.exit(0)

    records: list[dict] = []
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if args.phase and r.get("phase") != args.phase:
                continue
            records.append(r)

    if not records:
        print("No records found (or no records match the phase filter).")
        return

    total_cost = sum(r["cost_usd"] for r in records)
    unique_runs = len({r["transaction_id"] for r in records})
    by_agent: dict = defaultdict(lambda: {"cost": 0.0, "calls": 0})
    by_phase: dict = defaultdict(float)
    by_tier: dict = defaultdict(int)
    breaches = [r for r in records if r.get("budget_breached")]

    for r in records:
        by_agent[r["agent_name"]]["cost"] += r["cost_usd"]
        by_agent[r["agent_name"]]["calls"] += 1
        by_phase[r.get("phase", "unknown")] += r["cost_usd"]
        by_tier[r.get("tier", "unknown")] += 1

    print(f"\n{'='*55}")
    print(f"  COST REPORT  |  {len(records)} calls across {unique_runs} transactions")
    print(f"{'='*55}")
    print(f"\n  Total spend:             ${total_cost:.5f}")
    avg_per_tx = total_cost / unique_runs if unique_runs else 0
    print(f"  Avg cost per transaction: ${avg_per_tx:.5f}")
    print(f"  Projected/100 tx:         ${avg_per_tx * 100:.3f}")
    print(f"  Budget breaches:          {len(breaches)}")

    print(f"\n  --- By Phase ---")
    for phase, cost in sorted(by_phase.items()):
        print(f"    {phase}: ${cost:.5f}")

    print(f"\n  --- By Agent (avg cost/call) ---")
    for agent, data in sorted(by_agent.items()):
        avg = data["cost"] / data["calls"] if data["calls"] else 0
        print(f"    {agent}: ${avg:.6f}/call  ({data['calls']} calls)")

    total_calls = len(records)
    cheap_pct = by_tier.get("cheap", 0) / total_calls * 100 if total_calls else 0
    print(f"\n  --- Model Routing Efficiency ---")
    print(f"    Cheap tier:  {cheap_pct:.1f}%  (ExplanationAgent)")
    print(f"    Strong tier: {100-cheap_pct:.1f}%  (EvalAgent)")
    print(f"\n{'='*55}\n")


if __name__ == "__main__":
    main()
