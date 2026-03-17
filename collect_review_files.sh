#!/usr/bin/env bash
# =============================================================================
# collect_review_files.sh
# Collects files requested by external reviewer into a single ZIP archive.
# Run from inside: ~/projects/AI-Engineering/fraud-explanation-eval/
# Usage: bash collect_review_files.sh
# =============================================================================

set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m'

log()  { echo -e "${GREEN}[collect]${NC} $1"; }
warn() { echo -e "${YELLOW}[skip]${NC} $1"; }
info() { echo -e "${CYAN}[info]${NC} $1"; }

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="/tmp/review_files_${TIMESTAMP}"
ARCHIVE="review_files_${TIMESTAMP}.zip"

mkdir -p "$OUTPUT_DIR"

# =============================================================================
# Helper: copy a file preserving its relative path, warn if missing
# =============================================================================
collect() {
    local src="$1"
    local label="$2"
    if [ -f "$src" ]; then
        local dest_dir="$OUTPUT_DIR/$(dirname "$src")"
        mkdir -p "$dest_dir"
        cp "$src" "$dest_dir/"
        log "Collected: $src  ($label)"
    else
        warn "Not found:  $src  ($label)"
    fi
}

collect_dir() {
    local src_dir="$1"
    local label="$2"
    if [ -d "$src_dir" ] && [ "$(ls -A "$src_dir" 2>/dev/null)" ]; then
        cp -r "$src_dir" "$OUTPUT_DIR/$(dirname "$src_dir")/"
        log "Collected dir: $src_dir  ($label)"
    else
        warn "Empty/missing dir: $src_dir  ($label)"
    fi
}

echo ""
info "Collecting files for external review..."
info "Project root: $(pwd)"
echo ""

# =============================================================================
# GROUP 1: Runtime-generated values (highest analytical value)
# =============================================================================
log "--- Group 1: Runtime-generated files ---"

collect "cost_log.jsonl"          "LLM call audit trail — confirms cost tracking is wired"
collect "docs/PHASE_LOG.md"       "Actual gate verdicts per phase"
collect "CLAUDE.md"               "Current phase line + non-negotiable rules"

# =============================================================================
# GROUP 2: Files that may have diverged from scaffold spec
# =============================================================================
log "--- Group 2: Spec and schema files ---"

collect "docs/specs/DETECTOR_SPEC.md"     "Detector spec — check for drift"
collect "docs/specs/EXPLANATION_SPEC.md"  "Explanation spec — check for drift"
collect "docs/specs/EVAL_SPEC.md"         "Eval spec — check for drift"
collect "docs/ADR.md"                     "Architecture decision records"
collect "docs/COST_BUDGET.md"             "Cost budget with Phase Development Cost Log"
collect "src/schemas/detection.py"        "FraudDetectionResult schema + validators"
collect "src/schemas/explanation.py"      "ExplanationResult schema + validators"

# =============================================================================
# GROUP 3: ML artifacts
# =============================================================================
log "--- Group 3: ML artifacts ---"

collect "models/artifacts/version.txt"    "Trained model version"

# model.json is large — include a summary instead
if [ -f "models/artifacts/model.json" ]; then
    SIZE=$(du -sh models/artifacts/model.json | cut -f1)
    echo "model.json exists (size: $SIZE)" > "$OUTPUT_DIR/models/artifacts/model_json_summary.txt"
    log "Summarised: models/artifacts/model.json  (${SIZE} — not included in ZIP, confirmed present)"
else
    warn "Not found: models/artifacts/model.json"
fi

if [ -f "models/artifacts/model.features.json" ]; then
    mkdir -p "$OUTPUT_DIR/models/artifacts"
    cp "models/artifacts/model.features.json" "$OUTPUT_DIR/models/artifacts/"
    log "Collected: models/artifacts/model.features.json  (feature names used by model)"
fi

# =============================================================================
# GROUP 4: Test output files
# =============================================================================
log "--- Group 4: Test outputs ---"

collect "evals/EVAL_RESULTS.md"   "Golden scenario results from live EvalAgent run"
collect ".coverage"               "Raw coverage data"

# Generate a readable coverage report if possible
if [ -f ".coverage" ] && command -v python3 &>/dev/null; then
    python3 -m coverage report 2>/dev/null > "$OUTPUT_DIR/coverage_report.txt" && \
        log "Generated: coverage_report.txt  (human-readable coverage summary)" || \
        warn "Could not generate coverage report from .coverage file"
fi

# htmlcov summary — copy index only (not all 295 HTML files)
if [ -f "htmlcov/index.html" ]; then
    mkdir -p "$OUTPUT_DIR/htmlcov"
    cp "htmlcov/index.html" "$OUTPUT_DIR/htmlcov/"
    log "Collected: htmlcov/index.html  (coverage HTML index)"
else
    warn "Not found: htmlcov/index.html — run 'make coverage' first"
fi

# =============================================================================
# GROUP 5: Additional context files useful for review
# =============================================================================
log "--- Group 5: Additional context ---"

collect "docs/SECURITY.md"        "Populated security doc with test references"
collect "pyproject.toml"          "Dependency versions and test config"
collect "Makefile"                "All make targets including advance-phase"

# =============================================================================
# Write manifest
# =============================================================================
MANIFEST="$OUTPUT_DIR/MANIFEST.txt"
{
    echo "Review Files — fraud-explanation-eval"
    echo "Collected: $(date)"
    echo "Project:   $(pwd)"
    echo ""
    echo "FILES INCLUDED:"
    find "$OUTPUT_DIR" -type f | sort | sed "s|$OUTPUT_DIR/||" | while read -r f; do
        SIZE=$(du -sh "$OUTPUT_DIR/$f" 2>/dev/null | cut -f1)
        echo "  $SIZE  $f"
    done
    echo ""
    echo "FILES SKIPPED (see warnings above):"
    echo "  models/artifacts/model.json  (too large for upload — confirmed present)"
    echo "  data/raw/                    (gitignored — 677MB of IEEE-CIS CSVs)"
    echo "  data/processed/              (gitignored — generated parquet files)"
} > "$MANIFEST"

log "Written: MANIFEST.txt"

# =============================================================================
# Create ZIP
# =============================================================================
log "Creating archive: $ARCHIVE"
cd /tmp
zip -qr "$ARCHIVE" "review_files_${TIMESTAMP}/"
mv "$ARCHIVE" "$(dirs +1 2>/dev/null || echo "$OLDPWD")/" 2>/dev/null || true

# Find where it ended up
FINAL_PATH=""
if [ -f "$(pwd)/$ARCHIVE" ]; then
    FINAL_PATH="$(pwd)/$ARCHIVE"
elif [ -f "/tmp/$ARCHIVE" ]; then
    FINAL_PATH="/tmp/$ARCHIVE"
    cp "/tmp/$ARCHIVE" "$OLDPWD/"
    FINAL_PATH="$OLDPWD/$ARCHIVE"
fi

cd - > /dev/null

# =============================================================================
# Summary
# =============================================================================
echo ""
echo -e "${GREEN}══════════════════════════════════════════${NC}"
echo -e "${GREEN}  Collection complete!${NC}"
echo -e "${GREEN}══════════════════════════════════════════${NC}"
echo ""
echo "  Archive: $(pwd)/$ARCHIVE"
echo "  Size:    $(du -sh "$(pwd)/$ARCHIVE" 2>/dev/null | cut -f1 || echo 'see /tmp')"
echo ""
echo "  Files collected:"
find "$OUTPUT_DIR" -type f | grep -v MANIFEST | sort | sed "s|$OUTPUT_DIR/||" | \
    while read -r f; do echo "    $f"; done
echo ""
echo "  Upload this ZIP to the reviewer."
echo ""

# Cleanup temp dir
rm -rf "$OUTPUT_DIR"
