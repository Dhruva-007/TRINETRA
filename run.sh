#!/usr/bin/env bash
# =============================================================================
# TRINETRA — run.sh
# Integrated pipeline runner: telemetry extraction + ML inference
#
# All paths are resolved relative to this script's directory (the repo root),
# so this script works on any machine that has cloned the repository.
#
# Usage (run from repo root):
#   ./run.sh                          # full pipeline (requires Zeek + PCAP)
#   ./run.sh --skip-telemetry         # ML only, use existing data/processed/labeled-conn.csv
#   ./run.sh /path/to/zeek.csv        # score a specific Zeek CSV
#   ./run.sh --validate-only          # check ML artefacts and exit
#
# Expected repository layout after git clone:
#   ml/src/            ← ML Python modules
#   models/            ← trained model artefacts
#   data/processed/    ← input CSV (sample committed)
#   data/events/       ← output: security_events.jsonl (runtime, gitignored)
#   results/           ← output: trinetra_output.csv   (runtime, gitignored)
#   src/               ← integration scripts (ml_integration.py)
#   scripts/           ← telemetry pipeline scripts
#
# Exit codes:
#   0  success
#   1  pipeline failure (check output above)
#   2  bad arguments
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$SCRIPT_DIR/src"
ML_SRC_DIR="$SCRIPT_DIR/ml/src"
MODELS_DIR="$SCRIPT_DIR/models"
EVENTS_DIR="$SCRIPT_DIR/data/events"
RESULTS_DIR="$SCRIPT_DIR/results"
DEFAULT_CSV="$SCRIPT_DIR/data/processed/labeled-conn.csv"

VALIDATE_ONLY=false
SKIP_TELEMETRY=false
ZEEK_CSV="$DEFAULT_CSV"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --validate-only)  VALIDATE_ONLY=true;  shift ;;
        --skip-telemetry) SKIP_TELEMETRY=true; shift ;;
        --help | -h)
            echo "Usage: $0 [zeek.csv] [--validate-only] [--skip-telemetry]"
            exit 0 ;;
        -*)
            echo "[ERROR] Unknown flag: $1" >&2
            exit 2 ;;
        *)
            ZEEK_CSV="$1"
            shift ;;
    esac
done

echo ""
echo "================================================================"
echo "  TRINETRA — Integrated Pipeline"
echo "================================================================"
echo "  Repo root       : $SCRIPT_DIR"
echo "  ML source       : $ML_SRC_DIR"
echo "  Models          : $MODELS_DIR"
echo "  Input CSV       : $ZEEK_CSV"
echo "  Events output   : $EVENTS_DIR/security_events.jsonl"
echo ""

# ---------------------------------------------------------------------------
# Step 0 — Python environment
# ---------------------------------------------------------------------------
echo "[0/4] Checking Python environment ..."

if ! command -v python3 &>/dev/null; then
    echo "[ERROR] python3 not found." >&2
    exit 1
fi
PYTHON="python3"

$PYTHON -c "import pandas, numpy, sklearn, joblib, torch" 2>/dev/null || {
    echo "[ERROR] Missing Python packages. Install with:" >&2
    echo "          pip install -r $SCRIPT_DIR/requirements.txt" >&2
    exit 1
}
echo "  Python environment OK."

# ---------------------------------------------------------------------------
# Step 1 — Validate ML artefacts
# ---------------------------------------------------------------------------
echo ""
echo "[1/4] Validating ML artefacts ..."

$PYTHON "$SRC_DIR/ml_integration.py" --validate-only || {
    echo "" >&2
    echo "[ERROR] Artefact validation failed. See output above." >&2
    echo "  Artefacts should be present in: $MODELS_DIR" >&2
    echo "  If missing, regenerate with:" >&2
    echo "    python3 ml/src/train_isolation_forest.py" >&2
    echo "    python3 ml/src/train_gru.py" >&2
    echo "    python3 ml/src/trinetra_pipeline.py --build-config" >&2
    exit 1
}

if [ "$VALIDATE_ONLY" = true ]; then
    echo "  --validate-only: stopping after artefact check."
    exit 0
fi

# ---------------------------------------------------------------------------
# Step 2 — Telemetry pipeline (optional)
# ---------------------------------------------------------------------------
echo ""
echo "[2/4] Telemetry pipeline ..."

if [ "$SKIP_TELEMETRY" = true ]; then
    echo "  --skip-telemetry: using existing CSV: $ZEEK_CSV"
else
    TELEM_SCRIPT="$SCRIPT_DIR/scripts/run_pipeline.sh"
    if [ -f "$TELEM_SCRIPT" ]; then
        echo "  Running telemetry pipeline ..."
        bash "$TELEM_SCRIPT" || {
            echo "  [WARNING] Telemetry pipeline failed (Zeek/PCAP may be unavailable)."
            echo "  Falling back to existing CSV: $ZEEK_CSV"
        }
    else
        echo "  No telemetry pipeline script found. Using existing CSV."
    fi
fi

if [ ! -f "$ZEEK_CSV" ]; then
    echo "[ERROR] Input CSV not found: $ZEEK_CSV" >&2
    echo "  Run with --skip-telemetry and ensure data/processed/labeled-conn.csv exists." >&2
    exit 1
fi
echo "  Input CSV ready: $ZEEK_CSV"

# ---------------------------------------------------------------------------
# Step 3 — ML inference
# ---------------------------------------------------------------------------
echo ""
echo "[3/4] Running ML inference (IF + GRU + Threat Index) ..."

mkdir -p "$EVENTS_DIR" "$RESULTS_DIR"

$PYTHON "$SRC_DIR/ml_integration.py" \
    "$ZEEK_CSV" \
    --events-out "$EVENTS_DIR/security_events.jsonl" \
    --ml-output  "$RESULTS_DIR/trinetra_output.csv" || {
    echo "[ERROR] ML inference failed. See output above." >&2
    exit 1
}

# ---------------------------------------------------------------------------
# Step 4 — Evaluator report (optional)
# ---------------------------------------------------------------------------
echo ""
echo "[4/4] Generating evaluator report ..."

PRESENT_SCRIPT="$SCRIPT_DIR/ml/src/make_presentable.py"
if [ ! -f "$PRESENT_SCRIPT" ]; then
    # Fallback: look in repo root
    PRESENT_SCRIPT="$SCRIPT_DIR/make_presentable.py"
fi

if [ -f "$PRESENT_SCRIPT" ] && [ -f "$RESULTS_DIR/trinetra_output.csv" ]; then
    $PYTHON "$PRESENT_SCRIPT" || echo "  [WARNING] make_presentable.py failed (non-fatal)."
else
    echo "  make_presentable.py not found or no output CSV — skipping."
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "================================================================"
echo "  TRINETRA — Pipeline Complete"
echo "================================================================"

if [ -f "$EVENTS_DIR/security_events.jsonl" ]; then
    N_EVENTS=$(wc -l < "$EVENTS_DIR/security_events.jsonl")
    echo "  Security events : $N_EVENTS"
    echo "  Events file     : $EVENTS_DIR/security_events.jsonl"
fi
if [ -f "$RESULTS_DIR/trinetra_output.csv" ]; then
    N_WIN=$(tail -n +2 "$RESULTS_DIR/trinetra_output.csv" | wc -l)
    echo "  ML output       : $N_WIN windows  ($RESULTS_DIR/trinetra_output.csv)"
fi
echo ""
