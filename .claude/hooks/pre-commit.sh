#!/usr/bin/env bash
set -e
echo "[pre-commit] Running lint..."
make lint || { echo "BLOCKED: lint failed. Fix before committing."; exit 1; }
echo "[pre-commit] Running tests..."
make test-fast || { echo "BLOCKED: tests failing. Fix before committing."; exit 1; }
echo "[pre-commit] Checking for zero-cost records in cost_log.jsonl..."
if [ -f cost_log.jsonl ]; then
  ZERO_COST=$(python3 -c "
import json
count = 0
with open('cost_log.jsonl') as f:
    for line in f:
        r = json.loads(line)
        if r.get('cost_usd', 1) == 0.0:
            count += 1
print(count)
" 2>/dev/null || echo "0")
  if [ "$ZERO_COST" -gt "0" ]; then
    echo "BLOCKED: $ZERO_COST records with cost_usd=0.0 in cost_log.jsonl"
    echo "All LLM calls must use record_agent_call() with real token counts."
    exit 1
  fi
fi
echo "[pre-commit] All gates passed."
