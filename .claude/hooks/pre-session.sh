#!/usr/bin/env bash
echo "=== SESSION START ==="
if command -v make &> /dev/null; then
  make status 2>/dev/null || echo "[warn] make status failed — run manually"
fi
echo ""
echo "Current phase: $(grep 'Current Phase' CLAUDE.md | head -1)"
echo ""
echo "Protocol reminder:"
echo "  1. CI must be GREEN before feature work"
echo "  2. /start-phase → confirmed, proceed → implement → /agent-review → commit"
echo "  3. make advance-phase PHASE='N — Name' before /clear"
echo "  4. Every LLM call must append to cost_log.jsonl"
