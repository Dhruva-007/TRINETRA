"""
TRINETRA -- trinetra_pipeline.py
==================================
End-to-end ML integration layer.

Architecture
------------
    raw Zeek data (CSV or DataFrame)
            |
      canonical telemetry
            |
      +-----+-----+
      |           |
  IF inference  GRU inference
      |           |
  anomaly_score  trajectory_deviation
      |           |
  IF normalization  GRU normalization
      |           |
  if_signal [0,1]  gru_signal [0,1]
      +-----+-----+
            |
      Threat Index = 0.6*if_signal + 0.4*gru_signal
            |
      severity + evidence
            |
      unified structured output

Key design decisions
---------------------
* IF operates at CONNECTION level; GRU at 10-SECOND WINDOW level.
* Temporal alignment: for each GRU window, aggregate IF flow-level
  scores (mean anomaly_score_raw) within the same 10-second window.
* Normalization uses percentile clipping against a reference
  distribution saved at integration-config build time.
* Threat Index weights are configurable (default 0.6 / 0.4).
* Labels NEVER enter live inference.

All imports are at the top of the file.
"""

import os
import sys
import json
import logging

import numpy as np
import pandas as pd
import joblib
import torch

# Insert src/ onto path for sibling imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from inference import (
    run_inference_on_dataframe as if_infer,
    load_model_artefacts as if_load,
    validate_artefact_consistency as if_validate,
)
from gru_inference import (
    predict_trajectory as gru_predict,
    load_gru_artefacts,
)
from gru_preprocessing import (
    aggregate_into_windows,
    AGGREGATION_WINDOW_SECONDS,
)
from data_preprocessing import infer_and_cast_types

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IF_MODEL_PATH      = os.path.join(PROJECT_ROOT, "models", "isolation_forest.joblib")
IF_PREPROCESSOR    = os.path.join(PROJECT_ROOT, "models", "preprocessor.joblib")
IF_SCHEMA_PATH     = os.path.join(PROJECT_ROOT, "models", "feature_schema.joblib")

GRU_MODEL_PATH     = os.path.join(PROJECT_ROOT, "models", "gru_trajectory.pt")
GRU_SCALER_PATH    = os.path.join(PROJECT_ROOT, "models", "gru_scaler.joblib")
GRU_CONFIG_PATH    = os.path.join(PROJECT_ROOT, "models", "gru_config.joblib")

THREAT_CONFIG_PATH = os.path.join(PROJECT_ROOT, "models", "threat_index_config.joblib")
OUTPUT_CSV_PATH    = os.path.join(PROJECT_ROOT, "results", "trinetra_output.csv")
PROCESSED_CSV      = os.path.join(PROJECT_ROOT, "data", "processed", "labeled-conn.csv")

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
# Constants
# ---------------------------------------------------------------------------

# Threat Index formula weights (not statistically optimised)
DEFAULT_IF_WEIGHT  = 0.6
DEFAULT_GRU_WEIGHT = 0.4

# Severity bands (operational visualization only — NOT probabilities)
SEVERITY_BANDS = [
    (0.80, 1.01, "Critical"),
    (0.60, 0.80, "High"),
    (0.30, 0.60, "Moderate"),
    (0.00, 0.30, "Low"),
]

# Forbidden columns that must never enter live inference
_FORBIDDEN = {"label", "detailedlabel", "anomaly_score", "predicted_anomaly",
              "anomaly_score_raw", "anomaly_score_negated", "if_signal",
              "gru_signal", "threat_index", "severity"}

# Model version tag (bumped when models are retrained)
MODEL_VERSION = "IF-v1.GRU-v1.TI-v1"

# ---------------------------------------------------------------------------
# Severity helper
# ---------------------------------------------------------------------------

def _assign_severity(ti: float) -> str:
    for lo, hi, label in SEVERITY_BANDS:
        if lo <= ti < hi:
            return label
    return "Low"


# ---------------------------------------------------------------------------
# Threat Index config builder
# ---------------------------------------------------------------------------

def build_threat_index_config(
        normal_if_scores: np.ndarray,
        normal_gru_deviations: np.ndarray,
        if_weight: float = DEFAULT_IF_WEIGHT,
        gru_weight: float = DEFAULT_GRU_WEIGHT,
        clip_percentile: float = 95.0,
        save_path: str = THREAT_CONFIG_PATH,
) -> dict:
    """
    Build and persist the normalization reference configuration.

    Normalization method (percentile clipping):
        signal = clip(raw_value / reference_p95, 0, 1)

    Reference values are derived from NORMAL traffic scores only.
    Labels are not used to determine the reference.

    IF: anomaly_score_negated (higher = more anomalous)
    GRU: trajectory_deviation (Euclidean distance)

    Parameters
    ----------
    normal_if_scores       : 1-D array of IF anomaly_score_negated on normal data
    normal_gru_deviations  : 1-D array of GRU trajectory_deviation on normal data
    if_weight              : Threat Index weight for IF signal
    gru_weight             : Threat Index weight for GRU signal
    clip_percentile        : percentile used as the upper reference bound

    Returns the config dict.
    """
    if len(normal_if_scores) == 0:
        raise ValueError("normal_if_scores must be non-empty.")
    if len(normal_gru_deviations) == 0:
        raise ValueError("normal_gru_deviations must be non-empty.")

    # Reference bounds
    if_ref_min   = float(np.percentile(normal_if_scores, 0))
    if_ref_max   = float(np.percentile(normal_if_scores, clip_percentile))
    gru_ref_min  = float(np.percentile(normal_gru_deviations, 0))
    gru_ref_max  = float(np.percentile(normal_gru_deviations, clip_percentile))

    # Guard against degenerate reference (min==max)
    if if_ref_max <= if_ref_min:
        logger.warning("IF reference max<=min; setting range to 1.0 for safety.")
        if_ref_max = if_ref_min + 1.0
    if gru_ref_max <= gru_ref_min:
        logger.warning("GRU reference max<=min; setting range to 1.0 for safety.")
        gru_ref_max = gru_ref_min + 1.0

    config = {
        "if_weight":          if_weight,
        "gru_weight":         gru_weight,
        "clip_percentile":    clip_percentile,
        "if_ref_min":         if_ref_min,
        "if_ref_max":         if_ref_max,
        "gru_ref_min":        gru_ref_min,
        "gru_ref_max":        gru_ref_max,
        "n_normal_if":        int(len(normal_if_scores)),
        "n_normal_gru":       int(len(normal_gru_deviations)),
        "severity_bands":     SEVERITY_BANDS,
        "model_version":      MODEL_VERSION,
        "note": (
            "Normalization reference built from normal (Background+Benign) traffic only. "
            "Weights are operational, not statistically optimised. "
            "Signals are NOT probabilities."
        ),
    }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(config, save_path)
    logger.info("Threat Index config saved: %s", save_path)
    logger.info("  IF   ref [%.4f, %.4f]  (p%.0f)",
                if_ref_min, if_ref_max, clip_percentile)
    logger.info("  GRU  ref [%.4f, %.4f]  (p%.0f)",
                gru_ref_min, gru_ref_max, clip_percentile)
    logger.info("  Weights: IF=%.2f  GRU=%.2f", if_weight, gru_weight)
    return config


def load_threat_index_config(path: str = THREAT_CONFIG_PATH) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Threat Index config not found: {path}\n"
            "Run build_threat_index_config() or trinetra_pipeline.py --build-config first."
        )
    cfg = joblib.load(path)
    logger.info("Threat Index config loaded from: %s", path)
    return cfg


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def _normalize_signal(values: np.ndarray, ref_min: float, ref_max: float) -> np.ndarray:
    """
    Min-max normalize against the reference range then clip to [0, 1].

    signal = clip((value - ref_min) / (ref_max - ref_min), 0, 1)

    Values below ref_min → 0 (very normal)
    Values above ref_max → 1 (strongly anomalous)
    """
    span = ref_max - ref_min
    if span <= 0:
        span = 1.0
    normed = (values - ref_min) / span
    return np.clip(normed, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Temporal alignment
# ---------------------------------------------------------------------------

def align_if_to_gru_windows(
        if_result_df: pd.DataFrame,
        gru_window_seconds: int = AGGREGATION_WINDOW_SECONDS,
        ts_col: str = "ts",
) -> pd.DataFrame:
    """
    Aggregate connection-level IF scores into 10-second windows that match
    the GRU's temporal resolution.

    For each window:
        window_start        : left edge of window (seconds)
        mean_anomaly_score_negated : mean of IF anomaly_score_negated
        max_anomaly_score_negated  : max  (worst-case flow)
        n_flows             : number of connections in the window
        n_flagged           : flows where predicted_anomaly == 1

    anomaly_score_negated is used because it is directionally consistent:
        higher → more anomalous

    Returns a DataFrame indexed by window_start.
    """
    df = if_result_df.copy()
    if ts_col not in df.columns:
        raise ValueError(f"Column '{ts_col}' required for temporal alignment.")
    df[ts_col] = pd.to_numeric(df[ts_col], errors="coerce")
    df = df.dropna(subset=[ts_col])
    if df.empty:
        raise ValueError("No valid timestamps after casting.")

    ts_min = df[ts_col].min()
    df["_win_idx"] = ((df[ts_col] - ts_min) / gru_window_seconds).astype(int)
    df["_win_start"] = ts_min + df["_win_idx"] * gru_window_seconds

    grp = df.groupby("_win_start")
    records = []
    for win_start, g in grp:
        sc = g["anomaly_score_negated"].dropna()
        records.append({
            "window_start":                float(win_start),
            "mean_anomaly_score_negated":  float(sc.mean()) if len(sc) > 0 else 0.0,
            "max_anomaly_score_negated":   float(sc.max())  if len(sc) > 0 else 0.0,
            "n_flows":                     int(len(g)),
            "n_flagged":                   int((g["predicted_anomaly"] == 1).sum())
                                           if "predicted_anomaly" in g.columns else 0,
        })
    aligned = pd.DataFrame(records).sort_values("window_start").reset_index(drop=True)
    logger.info("IF scores aligned to %d windows (%d-s resolution).",
                len(aligned), gru_window_seconds)
    return aligned


def merge_signals(
        if_windows: pd.DataFrame,
        gru_result: pd.DataFrame,
        tolerance_seconds: float = 5.0,
) -> pd.DataFrame:
    """
    Merge IF window aggregates and GRU window results on window_start.

    Uses a nearest-timestamp merge with tolerance = tolerance_seconds/2.
    Windows with no counterpart in one source get NaN for that source's
    signal; these are handled downstream.

    Returns merged DataFrame with:
        window_start
        mean_anomaly_score_negated  (from IF)
        max_anomaly_score_negated   (from IF)
        n_flows, n_flagged          (from IF)
        trajectory_deviation        (from GRU)
        label                       (from GRU, for evaluation context)
    """
    if_df  = if_windows.copy()
    gru_df = gru_result[["window_start", "trajectory_deviation",
                          "label"]].copy() if "label" in gru_result.columns \
             else gru_result[["window_start", "trajectory_deviation"]].copy()

    # Merge on exact window_start after rounding to nearest window boundary
    # Use merge_asof (sorted nearest join)
    if_df  = if_df.sort_values("window_start")
    gru_df = gru_df.sort_values("window_start")

    merged = pd.merge_asof(
        gru_df,
        if_df,
        on="window_start",
        direction="nearest",
        tolerance=tolerance_seconds,
    )

    n_total   = len(merged)
    n_no_if   = merged["mean_anomaly_score_negated"].isna().sum()
    n_no_gru  = merged["trajectory_deviation"].isna().sum()
    logger.info("Signal merge: %d windows | %d missing IF | %d missing GRU",
                n_total, n_no_if, n_no_gru)
    return merged


# ---------------------------------------------------------------------------
# Threat Index calculation
# ---------------------------------------------------------------------------

def compute_threat_index(
        merged: pd.DataFrame,
        ti_config: dict,
) -> pd.DataFrame:
    """
    For each aligned window compute:
        if_signal   = normalized IF score ∈ [0,1]
        gru_signal  = normalized GRU deviation ∈ [0,1]
        threat_index = if_weight * if_signal + gru_weight * gru_signal
        severity    = severity band label

    Missing signals are filled with 0.0 (most conservative assumption).
    """
    df = merged.copy()

    if_scores  = df["mean_anomaly_score_negated"].fillna(0.0).values
    gru_devs   = df["trajectory_deviation"].fillna(0.0).values

    if_signal  = _normalize_signal(if_scores,
                                    ti_config["if_ref_min"],
                                    ti_config["if_ref_max"])
    gru_signal = _normalize_signal(gru_devs,
                                    ti_config["gru_ref_min"],
                                    ti_config["gru_ref_max"])

    w_if  = ti_config["if_weight"]
    w_gru = ti_config["gru_weight"]
    ti    = np.clip(w_if * if_signal + w_gru * gru_signal, 0.0, 1.0)

    df["if_signal"]    = np.round(if_signal,  4)
    df["gru_signal"]   = np.round(gru_signal, 4)
    df["threat_index"] = np.round(ti, 4)
    df["severity"]     = [_assign_severity(v) for v in ti]

    # Validation: all signals in [0,1]
    assert (if_signal  >= 0).all() and (if_signal  <= 1).all(), "IF signal out of [0,1]"
    assert (gru_signal >= 0).all() and (gru_signal <= 1).all(), "GRU signal out of [0,1]"
    assert (ti         >= 0).all() and (ti         <= 1).all(), "Threat Index out of [0,1]"

    logger.info(
        "Threat Index computed for %d windows. "
        "mean=%.4f  max=%.4f  severity counts: %s",
        len(df), ti.mean(), ti.max(),
        df["severity"].value_counts().to_dict(),
    )
    return df


# ---------------------------------------------------------------------------
# Unified output builder
# ---------------------------------------------------------------------------

def build_unified_output(ti_df: pd.DataFrame, model_version: str = MODEL_VERSION) -> pd.DataFrame:
    """
    Select and rename columns into the clean dashboard-ready schema.

    Output columns:
        window_start             : float, UNIX timestamp of window left edge
        n_flows                  : int, number of IF-scored flows in window
        n_flagged                : int, flows flagged by IF as anomalous
        mean_anomaly_score_negated : float, mean IF anomaly score (negated)
        trajectory_deviation     : float, GRU Euclidean deviation
        if_signal                : float ∈ [0,1]
        gru_signal               : float ∈ [0,1]
        threat_index             : float ∈ [0,1]
        severity                 : str (Low/Moderate/High/Critical)
        label                    : str (context only; not an ML input)
        model_version            : str
    """
    cols_needed = [
        "window_start",
        "n_flows", "n_flagged",
        "mean_anomaly_score_negated",
        "trajectory_deviation",
        "if_signal", "gru_signal",
        "threat_index", "severity",
    ]
    optional = ["label", "max_anomaly_score_negated"]

    df = ti_df.copy()
    for col in optional:
        if col not in df.columns:
            df[col] = None

    out_cols = cols_needed + [c for c in optional if c in df.columns]
    out = df[[c for c in out_cols if c in df.columns]].copy()
    out["model_version"] = model_version
    return out.sort_values("window_start").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Artefact loading & validation
# ---------------------------------------------------------------------------

def load_all_artefacts() -> tuple:
    """
    Load and cross-validate all ML artefacts at startup.

    Returns: (if_model, if_preprocessor, if_schema,
               gru_model, gru_scaler, gru_config,
               ti_config)

    Raises FileNotFoundError if any artefact is missing.
    Raises ValueError if dimensions are inconsistent.
    """
    logger.info("Loading TRINETRA ML artefacts ...")

    # IF artefacts
    if_model, if_preprocessor, if_schema = if_load()
    if_validate(if_preprocessor, if_schema)

    # GRU artefacts
    gru_model, gru_scaler, gru_config = load_gru_artefacts()

    # Threat Index config
    ti_config = load_threat_index_config()

    # Cross-validate IF schema vs GRU config (they should differ; log clearly)
    logger.info("IF final features: %d  |  GRU features: %d",
                len(if_schema["final_feature_names"]),
                gru_config["n_features"])
    logger.info("GRU aggregation window: %d s", gru_config["aggregation_window"])

    # Verify model version compatibility
    if ti_config.get("model_version") != MODEL_VERSION:
        logger.warning(
            "Threat Index config model_version=%s does not match current MODEL_VERSION=%s. "
            "Consider rebuilding the Threat Index config.",
            ti_config.get("model_version"), MODEL_VERSION
        )

    logger.info("All artefacts loaded and validated.")
    return (if_model, if_preprocessor, if_schema,
            gru_model, gru_scaler, gru_config,
            ti_config)


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def run_trinetra_pipeline(
        input_data,
        ti_config: dict = None,
        if_model=None, if_preprocessor=None, if_schema=None,
        gru_model=None, gru_scaler=None, gru_config=None,
        save_output: bool = False,
        output_path: str = OUTPUT_CSV_PATH,
) -> pd.DataFrame:
    """
    End-to-end TRINETRA ML pipeline.

    Parameters
    ----------
    input_data : str (CSV path) or pd.DataFrame
        Raw Zeek connection telemetry. Must contain 'ts' for temporal alignment.

    ti_config  : Threat Index config dict. If None, loaded from disk.
    if_model, if_preprocessor, if_schema : IF artefacts. If None, loaded from disk.
    gru_model, gru_scaler, gru_config    : GRU artefacts. If None, loaded from disk.

    save_output : bool — if True, write results to output_path.
    output_path : str  — path for output CSV.

    Returns
    -------
    pd.DataFrame with unified output schema (see build_unified_output).
    """
    # -----------------------------------------------------------------------
    # Input loading
    # -----------------------------------------------------------------------
    if isinstance(input_data, str):
        logger.info("Loading input from CSV: %s", input_data)
        df = pd.read_csv(input_data)
    elif isinstance(input_data, pd.DataFrame):
        df = input_data.copy()
    else:
        raise TypeError(f"input_data must be str (CSV path) or pd.DataFrame, got {type(input_data)}")

    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    # Safety: labels must not be used as live inference features
    live_forbidden = _FORBIDDEN.intersection(set(df.columns))
    # Labels are allowed in the DataFrame for evaluation context but must
    # never enter the ML feature matrices (enforced inside IF/GRU modules).
    logger.info("Input: %d rows × %d cols", len(df), len(df.columns))
    if live_forbidden - {"label", "detailedlabel"}:
        logger.warning("Input contains derived score columns: %s — they will be ignored.",
                       live_forbidden - {"label", "detailedlabel"})

    # -----------------------------------------------------------------------
    # Load artefacts on demand
    # -----------------------------------------------------------------------
    if any(x is None for x in [if_model, if_preprocessor, if_schema]):
        if_model, if_preprocessor, if_schema = if_load()
        if_validate(if_preprocessor, if_schema)

    if any(x is None for x in [gru_model, gru_scaler, gru_config]):
        gru_model, gru_scaler, gru_config = load_gru_artefacts()

    if ti_config is None:
        ti_config = load_threat_index_config()

    # -----------------------------------------------------------------------
    # Step 1: Isolation Forest inference (flow-level)
    # -----------------------------------------------------------------------
    logger.info("--- Step 1: IF inference ---")
    if_result = if_infer(df, if_model, if_preprocessor, if_schema)
    logger.info("IF scored %d flows.", len(if_result))

    # Verify no labels entered IF
    assert "label" not in if_schema["numerical_features"], "LEAK: label in IF features"

    # -----------------------------------------------------------------------
    # Step 2: GRU inference (window-level)
    # -----------------------------------------------------------------------
    logger.info("--- Step 2: GRU inference ---")
    gru_result = gru_predict(df, gru_model, gru_scaler, gru_config)
    logger.info("GRU scored %d windows.", len(gru_result))

    if gru_result.empty:
        raise ValueError("GRU returned empty results — check temporal coverage.")

    # -----------------------------------------------------------------------
    # Step 3: Temporal alignment
    # -----------------------------------------------------------------------
    logger.info("--- Step 3: Temporal alignment ---")
    if_windows = align_if_to_gru_windows(
        if_result,
        gru_window_seconds=gru_config["aggregation_window"],
    )
    merged = merge_signals(if_windows, gru_result)

    if merged.empty:
        raise ValueError("Signal merge produced empty result — no overlapping windows.")

    # -----------------------------------------------------------------------
    # Step 4: Signal normalization + Threat Index
    # -----------------------------------------------------------------------
    logger.info("--- Step 4: Threat Index computation ---")
    ti_df = compute_threat_index(merged, ti_config)

    # -----------------------------------------------------------------------
    # Step 5: Build unified output
    # -----------------------------------------------------------------------
    output = build_unified_output(ti_df, model_version=MODEL_VERSION)

    if save_output:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        output.to_csv(output_path, index=False)
        logger.info("Unified output saved: %s  (%d rows)", output_path, len(output))

    # Summary log
    severity_counts = output["severity"].value_counts().to_dict()
    logger.info(
        "Pipeline complete: %d windows | TI mean=%.4f max=%.4f | %s",
        len(output), output["threat_index"].mean(),
        output["threat_index"].max(), severity_counts,
    )
    return output


# ---------------------------------------------------------------------------
# Config builder mode (run once to calibrate normalization)
# ---------------------------------------------------------------------------

def build_config_from_data(
        csv_path: str = PROCESSED_CSV,
        if_weight: float = DEFAULT_IF_WEIGHT,
        gru_weight: float = DEFAULT_GRU_WEIGHT,
        clip_percentile: float = 95.0,
) -> dict:
    """
    Build the Threat Index calibration config from the labeled dataset.

    Uses NORMAL traffic (Background + Benign) only for the reference
    distribution. Labels are used only to select the reference subset.

    Steps:
      1. Run IF on the full dataset, keep normal-row scores.
      2. Run GRU on normal-only aggregated windows, keep deviations.
      3. Save threat_index_config.joblib.
    """
    logger.info("Building Threat Index config from: %s", csv_path)

    df = pd.read_csv(csv_path)
    if_model, if_preprocessor, if_schema = if_load()
    if_validate(if_preprocessor, if_schema)
    gru_model, gru_scaler, gru_config = load_gru_artefacts()

    # IF reference: normal flows only
    normal_mask = df["label"].isin(["Background", "Benign"])
    df_normal   = df[normal_mask].copy()
    if_result   = if_infer(df_normal, if_model, if_preprocessor, if_schema)
    normal_if_scores = if_result["anomaly_score_negated"].dropna().values
    logger.info("Normal IF scores: n=%d  mean=%.4f  p95=%.4f",
                len(normal_if_scores),
                float(np.mean(normal_if_scores)),
                float(np.percentile(normal_if_scores, clip_percentile)))

    # GRU reference: normal window deviations
    gru_result = gru_predict(df_normal, gru_model, gru_scaler, gru_config)
    normal_gru_devs = gru_result["trajectory_deviation"].dropna().values
    logger.info("Normal GRU deviations: n=%d  mean=%.4f  p95=%.4f",
                len(normal_gru_devs),
                float(np.mean(normal_gru_devs)),
                float(np.percentile(normal_gru_devs, clip_percentile)))

    ti_config = build_threat_index_config(
        normal_if_scores, normal_gru_devs,
        if_weight=if_weight,
        gru_weight=gru_weight,
        clip_percentile=clip_percentile,
    )
    return ti_config


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="TRINETRA -- End-to-end ML pipeline (IF + GRU + Threat Index)"
    )
    parser.add_argument(
        "input_path", nargs="?", default=PROCESSED_CSV,
        help="Path to Zeek CSV file (default: labeled-conn.csv)."
    )
    parser.add_argument(
        "--build-config", action="store_true",
        help="Build/rebuild the Threat Index normalization config from normal data."
    )
    parser.add_argument(
        "--output", "-o", default=OUTPUT_CSV_PATH,
        help="Path to write unified output CSV."
    )
    parser.add_argument(
        "--if-weight",  type=float, default=DEFAULT_IF_WEIGHT
    )
    parser.add_argument(
        "--gru-weight", type=float, default=DEFAULT_GRU_WEIGHT
    )
    args = parser.parse_args()

    if args.build_config:
        logger.info("=== Building Threat Index config ===")
        cfg = build_config_from_data(
            args.input_path,
            if_weight=args.if_weight,
            gru_weight=args.gru_weight,
        )
        logger.info("Config saved. Re-run without --build-config to run inference.")
    else:
        logger.info("=== TRINETRA End-to-End Pipeline ===")
        result = run_trinetra_pipeline(
            args.input_path,
            save_output=True,
            output_path=args.output,
        )
        print("\n" + "=" * 65)
        print("  TRINETRA -- Unified Threat Intelligence Summary")
        print("=" * 65)
        print(f"  Windows processed    : {len(result)}")
        print(f"  Threat Index mean    : {result['threat_index'].mean():.4f}")
        print(f"  Threat Index max     : {result['threat_index'].max():.4f}")
        print(f"  Severity breakdown   :")
        for sev, cnt in result["severity"].value_counts().items():
            print(f"    {sev:10s}: {cnt}")
        print(f"\n  Output saved to: {args.output}")
        print("\n--- Top 10 windows by Threat Index ---")
        top = result.sort_values("threat_index", ascending=False).head(10)
        display = ["window_start", "n_flows", "if_signal", "gru_signal",
                   "threat_index", "severity"]
        if "label" in top.columns:
            display.append("label")
        print(top[[c for c in display if c in top.columns]].to_string(index=False))
