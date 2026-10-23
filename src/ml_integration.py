"""
TRINETRA — src/ml_integration.py
==================================
ML integration entry point for the shared TRINETRA repository.

This module bridges the shared telemetry pipeline to the ML inference
layer.  All paths are resolved relative to the shared repository root,
so any team member who clones https://github.com/Dhruva-007/TRINETRA.git
can run the integration without any files outside the repository.

Directory layout (all paths inside the shared repo root):
  ml/src/                     ← ML source modules
    trinetra_pipeline.py      ← end-to-end pipeline
    inference.py              ← Isolation Forest inference
    gru_inference.py          ← GRU inference
    gru_preprocessing.py      ← temporal window aggregation
    feature_engineering.py    ← feature construction
    data_preprocessing.py     ← Zeek log → CSV
  models/                     ← trained model artefacts (committed)
    isolation_forest.joblib
    preprocessor.joblib
    feature_schema.joblib
    gru_trajectory.pt
    gru_scaler.joblib
    gru_config.joblib
    threat_index_config.joblib
  data/processed/             ← input Zeek CSV (sample committed)
    labeled-conn.csv
  data/events/                ← output security events (gitignored)
    security_events.jsonl
  results/                    ← ML output CSV (gitignored)
    trinetra_output.csv

Path contract:
  __file__ is …/src/ml_integration.py
  _REPO_ROOT = dirname(dirname(__file__))  →  shared repo root
  _ML_SRC    = _REPO_ROOT / ml / src      →  ML Python modules

The ML modules themselves resolve PROJECT_ROOT as
  dirname(dirname(__file__))  when imported from ml/src/
which also resolves to _REPO_ROOT, so models/, data/, results/ are
shared consistently between ml_integration and the ML modules.

No absolute paths. No dependencies outside this repository.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import sys

import pandas as pd

# ---------------------------------------------------------------------------
# Canonical path resolution — all relative to the shared repo root
# ---------------------------------------------------------------------------
# __file__   → …/TRINETRA/src/ml_integration.py
# _REPO_ROOT → …/TRINETRA/                        (shared repo root)
# _ML_SRC    → …/TRINETRA/ml/src/                 (ML Python modules)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ML_SRC    = os.path.join(_REPO_ROOT, "ml", "src")

if _ML_SRC not in sys.path:
    sys.path.insert(0, _ML_SRC)

# ---------------------------------------------------------------------------
# Required artefact paths (all inside the shared repo)
# ---------------------------------------------------------------------------

_MODELS_DIR = os.path.join(_REPO_ROOT, "models")

REQUIRED_ARTEFACTS: dict[str, str] = {
    "isolation_forest.joblib":    os.path.join(_MODELS_DIR, "isolation_forest.joblib"),
    "preprocessor.joblib":        os.path.join(_MODELS_DIR, "preprocessor.joblib"),
    "feature_schema.joblib":      os.path.join(_MODELS_DIR, "feature_schema.joblib"),
    "gru_trajectory.pt":          os.path.join(_MODELS_DIR, "gru_trajectory.pt"),
    "gru_scaler.joblib":          os.path.join(_MODELS_DIR, "gru_scaler.joblib"),
    "gru_config.joblib":          os.path.join(_MODELS_DIR, "gru_config.joblib"),
    "threat_index_config.joblib": os.path.join(_MODELS_DIR, "threat_index_config.joblib"),
}

# Default input Zeek CSV (sample committed to the repo)
DEFAULT_ZEEK_CSV = os.path.join(_REPO_ROOT, "data", "processed", "labeled-conn.csv")

# Outputs (both are gitignored — runtime only)
EVENTS_DIR    = os.path.join(_REPO_ROOT, "data", "events")
EVENTS_JSONL  = os.path.join(EVENTS_DIR, "security_events.jsonl")
ML_OUTPUT_CSV = os.path.join(_REPO_ROOT, "results", "trinetra_output.csv")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Artefact validation
# ---------------------------------------------------------------------------

def validate_artefacts() -> None:
    """
    Verify every required model artefact exists inside the repository.
    Raises FileNotFoundError listing all missing files on first failure.

    Call this before importing any ML module — it gives a clear error
    instead of a cryptic ImportError or joblib failure.
    """
    missing = [
        f"{name}  →  {path}"
        for name, path in REQUIRED_ARTEFACTS.items()
        if not os.path.exists(path)
    ]
    if missing:
        raise FileNotFoundError(
            "Missing required ML artefacts in " + _MODELS_DIR + ":\n"
            + "\n".join(f"  {m}" for m in missing)
            + "\n\nTo generate artefacts from scratch:\n"
              "  cd <repo-root>\n"
              "  pip install -r requirements.txt\n"
              "  python ml/src/data_preprocessing.py\n"
              "  python ml/src/train_isolation_forest.py\n"
              "  python ml/src/train_gru.py\n"
              "  python ml/src/trinetra_pipeline.py --build-config\n"
              "Artefacts are committed to the repository and should be "
              "present after a normal git clone."
        )
    logger.info(
        "Artefact check passed: %d/%d files present in %s",
        len(REQUIRED_ARTEFACTS), len(REQUIRED_ARTEFACTS), _MODELS_DIR,
    )


# ---------------------------------------------------------------------------
# Security event helpers
# ---------------------------------------------------------------------------

def _safe_int(val) -> "int | None":
    """Return int(val), or None if val is NaN/None (float columns from merged DF)."""
    if val is None:
        return None
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass
    return int(val)


def _ts_to_iso(ts: float) -> str:
    """Convert a UNIX timestamp float to an ISO-8601 UTC string."""
    try:
        return datetime.datetime.utcfromtimestamp(float(ts)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
    except (ValueError, OSError, OverflowError):
        return str(ts)


def _build_evidence(row: pd.Series) -> list:
    """Build human-readable evidence strings from one ML output row."""
    evidence = []
    n_flows   = _safe_int(row.get("n_flows"))   or 0
    n_flagged = _safe_int(row.get("n_flagged")) or 0
    if n_flows > 0:
        evidence.append(
            f"{n_flows} flows in window; {n_flagged} flagged anomalous by IF"
        )
    if_sig = row.get("if_signal")
    if pd.notna(if_sig):
        evidence.append(f"IF anomaly signal = {float(if_sig):.4f} (0=normal, 1=most anomalous)")
    gru_sig = row.get("gru_signal")
    if pd.notna(gru_sig):
        evidence.append(f"GRU trajectory signal = {float(gru_sig):.4f}")
    dev = row.get("trajectory_deviation")
    if pd.notna(dev):
        evidence.append(f"Raw trajectory deviation = {float(dev):.2f} (Euclidean distance)")
    raw_if = row.get("mean_anomaly_score_negated")
    if pd.notna(raw_if):
        evidence.append(f"Mean IF anomaly_score_negated = {float(raw_if):.6f}")
    mv = row.get("model_version")
    if pd.notna(mv):
        evidence.append(f"Model version: {mv}")
    return evidence


def build_security_events(ml_output: pd.DataFrame) -> list:
    """
    Convert each ML pipeline output row into a security event dict
    compatible with security_event_schema.json.

    Schema mapping
    --------------
    event_id          → "TRN-{NNNN:04d}" (1-indexed per run)
    window            → window_start (UNIX timestamp float)
    timestamp         → ISO-8601 UTC string
    source            → null  (IPs not available at window level)
    destination       → null
    event_type        → "network_behavior_shift"
    current_anomaly   → if_signal  [0, 1]
    trajectory_change → gru_signal [0, 1]
    future_risk       → gru_signal [0, 1]
    threat_index      → threat_index [0, 1]
    severity          → Low | Moderate | High | Critical
    evidence          → list[str]
    technique         → null  (ATT&CK mapping not implemented)

    Extended fields (not in base schema; stable for dashboard):
    anomaly_score_negated_mean, trajectory_deviation, n_flows, n_flagged,
    model_version
    """
    events = []
    for i, (_, row) in enumerate(ml_output.iterrows(), start=1):
        window_ts = row.get("window_start")
        event = {
            # Base schema fields
            "event_id":          f"TRN-{i:04d}",
            "window":            float(window_ts)       if pd.notna(window_ts) else None,
            "timestamp":         _ts_to_iso(window_ts) if pd.notna(window_ts) else None,
            "source":            None,
            "destination":       None,
            "event_type":        "network_behavior_shift",
            "current_anomaly":   float(row["if_signal"])    if pd.notna(row.get("if_signal"))    else None,
            "trajectory_change": float(row["gru_signal"])   if pd.notna(row.get("gru_signal"))   else None,
            "future_risk":       float(row["gru_signal"])   if pd.notna(row.get("gru_signal"))   else None,
            "threat_index":      float(row["threat_index"]) if pd.notna(row.get("threat_index")) else None,
            "evidence":          _build_evidence(row),
            "technique":         None,
            "severity":          str(row["severity"])       if pd.notna(row.get("severity"))      else None,
            # Extended ML fields (stable)
            "anomaly_score_negated_mean": (
                float(row["mean_anomaly_score_negated"])
                if pd.notna(row.get("mean_anomaly_score_negated")) else None
            ),
            "trajectory_deviation": (
                float(row["trajectory_deviation"])
                if pd.notna(row.get("trajectory_deviation")) else None
            ),
            "n_flows":       _safe_int(row.get("n_flows")),
            "n_flagged":     _safe_int(row.get("n_flagged")),
            "model_version": str(row["model_version"]) if pd.notna(row.get("model_version")) else None,
        }
        events.append(event)
    return events


def write_events_jsonl(events: list, path: str = EVENTS_JSONL) -> None:
    """Write security events as JSON Lines (one JSON object per line)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for event in events:
            fh.write(json.dumps(event, ensure_ascii=False) + "\n")
    logger.info("Security events written: %s  (%d events)", path, len(events))


# ---------------------------------------------------------------------------
# Main integration entry point
# ---------------------------------------------------------------------------

def run_ml_integration(
    zeek_csv: str = DEFAULT_ZEEK_CSV,
    events_path: str = EVENTS_JSONL,
    ml_output_path: str = ML_OUTPUT_CSV,
) -> pd.DataFrame:
    """
    Run the full ML pipeline on a Zeek connection CSV.

    Parameters
    ----------
    zeek_csv       : Path to input Zeek conn CSV (must contain Zeek fields).
    events_path    : Where to write security_events.jsonl (JSON Lines).
    ml_output_path : Where to write trinetra_output.csv.

    Returns
    -------
    pd.DataFrame with the ML pipeline output (one row per 10-second window).

    Raises
    ------
    FileNotFoundError  — missing artefacts or input CSV.
    ValueError         — empty or malformed input data.
    ImportError        — ML modules not importable (check ml/src/).
    """
    logger.info("=== TRINETRA ML Integration Layer ===")
    logger.info("Repo root : %s", _REPO_ROOT)
    logger.info("ML src    : %s", _ML_SRC)

    # 1. Validate artefacts before importing ML code
    validate_artefacts()

    # 2. Import ML pipeline (delayed so sys.path is set first)
    try:
        from trinetra_pipeline import run_trinetra_pipeline, load_all_artefacts
    except ImportError as exc:
        raise ImportError(
            f"Cannot import 'trinetra_pipeline' from {_ML_SRC}.\n"
            f"Cause: {exc}\n"
            "Ensure ml/src/ contains the ML source files and that all "
            "dependencies in requirements.txt are installed."
        ) from exc

    # 3. Validate input CSV
    if not os.path.exists(zeek_csv):
        raise FileNotFoundError(
            f"Input Zeek CSV not found: {zeek_csv}\n"
            "Either run the telemetry pipeline to generate one, or supply "
            "a valid path via the --input argument."
        )
    logger.info("Input CSV : %s", zeek_csv)

    # 4. Load artefacts and run inference
    logger.info("Loading ML artefacts from %s ...", _MODELS_DIR)
    (if_model, if_preprocessor, if_schema,
     gru_model, gru_scaler, gru_config,
     ti_config) = load_all_artefacts()

    logger.info("Running TRINETRA ML pipeline ...")
    os.makedirs(os.path.dirname(ml_output_path), exist_ok=True)
    ml_output = run_trinetra_pipeline(
        zeek_csv,
        if_model=if_model, if_preprocessor=if_preprocessor, if_schema=if_schema,
        gru_model=gru_model, gru_scaler=gru_scaler, gru_config=gru_config,
        ti_config=ti_config,
        save_output=True,
        output_path=ml_output_path,
    )

    # 5. Serialise security events
    logger.info("Building security events ...")
    events = build_security_events(ml_output)
    write_events_jsonl(events, events_path)

    # 6. Print summary
    severity_counts = ml_output["severity"].value_counts().to_dict()
    print("\n" + "=" * 65)
    print("  TRINETRA — ML Integration Complete")
    print("=" * 65)
    print(f"  Input CSV      : {zeek_csv}")
    print(f"  Windows scored : {len(ml_output)}")
    print(f"  TI mean        : {ml_output['threat_index'].mean():.4f}")
    print(f"  TI max         : {ml_output['threat_index'].max():.4f}")
    print("  Severity:")
    for sev in ["Critical", "High", "Moderate", "Low"]:
        print(f"    {sev:10s}: {severity_counts.get(sev, 0)}")
    print(f"\n  ML output CSV   : {ml_output_path}")
    print(f"  Security events : {events_path}  ({len(events)} events)")
    print("=" * 65)

    return ml_output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        prog="ml_integration",
        description=(
            "TRINETRA ML Integration — run IF + GRU + Threat Index on a Zeek CSV.\n"
            "All paths default to locations inside the repository."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "zeek_csv",
        nargs="?",
        default=DEFAULT_ZEEK_CSV,
        help="Path to Zeek connection CSV (default: data/processed/labeled-conn.csv)",
    )
    parser.add_argument(
        "--events-out", "-e",
        default=EVENTS_JSONL,
        metavar="PATH",
        help="Path to write security events JSONL (default: data/events/security_events.jsonl)",
    )
    parser.add_argument(
        "--ml-output", "-o",
        default=ML_OUTPUT_CSV,
        metavar="PATH",
        help="Path to write ML output CSV (default: results/trinetra_output.csv)",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate artefacts and exit without running inference.",
    )
    args = parser.parse_args()

    if args.validate_only:
        validate_artefacts()
        print("All ML artefacts present. System is ready.")
        sys.exit(0)

    try:
        run_ml_integration(
            zeek_csv=args.zeek_csv,
            events_path=args.events_out,
            ml_output_path=args.ml_output,
        )
        sys.exit(0)
    except (FileNotFoundError, ValueError, ImportError) as exc:
        logger.error("Integration failed: %s", exc)
        sys.exit(1)
