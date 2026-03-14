#!/usr/bin/env bash
# PostToolUse hook — checks for secrets after file writes
SUSPICIOUS=$(grep -rn "api_key\s*=\s*['\"][^$]" src/ --include="*.py" 2>/dev/null | grep -v ".env" | grep -v "os.getenv" | head -5 || true)
if [ -n "$SUSPICIOUS" ]; then
  echo "BLOCKED: Potential hardcoded secret detected:"
  echo "$SUSPICIOUS"
  echo "Use os.getenv() and .env only."
  exit 1
fi
