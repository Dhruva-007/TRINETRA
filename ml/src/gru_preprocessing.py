"""
TRINETRA -- gru_preprocessing.py
==================================
GRU Component, Step 1: Temporal network-state aggregation.

Pipeline:
  raw Zeek CSV (or DataFrame)
      |
      v
  feature engineering (reuses feature_engineering.py)
      |
      v
  sort chronologically by ts
      |
      v
  aggregate into 10-second time windows  -> network-state vectors
      |
      v
  sequence construction  (seq_len=5 sliding window)
      |
      v
  X.shape = (N, 5, F)   y.shape = (N, F)

Data contract
-------------
Required raw Zeek columns for feature engineering:
  orig_pkts, resp_pkts, orig_bytes, resp_bytes, duration,
  orig_ip_bytes, resp_ip_bytes, missed_bytes, id.orig_p, id.resp_p,
  proto, service, conn_state, history

GRU input features (all numerical, post-aggregation):
  duration_mean         mean of connection duration in window
  packet_count_sum      sum of packet_count in window
  total_bytes_sum       sum of total_bytes in window
  bytes_per_packet_mean mean bytes per packet in window
  packet_rate_mean      mean packet rate in window
  byte_rate_mean        mean byte rate in window
  sent_received_ratio_mean  mean sent/received ratio in window

Categorical features (proto, service) are NOT included in GRU because:
  - Proto is nearly constant (all ICMP/TCP/UDP; IF already handles it)
  - OHE categories inflate dimensionality with little temporal signal
  - Keeps GRU lightweight and explainable

Timestamp is used ONLY for sorting and windowing -- NOT as a GRU feature.
Labels are used ONLY for train/eval data selection -- NOT as GRU features.

All imports are at the top of the file.
"""

import os
import sys
import logging

import numpy as np
import pandas as pd

# Reuse existing feature engineering
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_engineering import engineer_features

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_CSV_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "labeled-conn.csv")

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
# GRU Feature Contract  -- canonical definition
# ---------------------------------------------------------------------------

# Aggregation window in seconds (chosen after inspecting timestamp distribution)
# Dataset spans 3008 s with ~992 rows; 10 s gives 174 non-empty normal windows
AGGREGATION_WINDOW_SECONDS = 10

# Sequence length (number of consecutive windows per GRU input)
SEQUENCE_LENGTH = 5

# GRU features: (source_column, aggregation_function, output_column_name)
# All are derived from the existing TRINETRA behavioural feature engineering.
GRU_FEATURE_SPEC = [
    ("duration",             "mean",  "duration_mean"),
    ("packet_count",         "sum",   "packet_count_sum"),
    ("total_bytes",          "sum",   "total_bytes_sum"),
    ("bytes_per_packet",     "mean",  "bytes_per_packet_mean"),
    ("packet_rate",          "mean",  "packet_rate_mean"),
    ("byte_rate",            "mean",  "byte_rate_mean"),
    ("sent_received_ratio",  "mean",  "sent_received_ratio_mean"),
]

# Final ordered GRU feature names (preserves exact order)
GRU_FEATURE_NAMES = [spec[2] for spec in GRU_FEATURE_SPEC]
N_GRU_FEATURES   = len(GRU_FEATURE_NAMES)

# Labels that must NEVER enter X or y
_FORBIDDEN_FEATURES = {"label", "detailedlabel", "anomaly_score",
                       "predicted_anomaly", "anomaly_score_raw",
                       "anomaly_score_negated", "ts"}

# ---------------------------------------------------------------------------
# Helper: type-cast without importing from data_preprocessing (standalone)
# ---------------------------------------------------------------------------

def _cast_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Cast known Zeek numeric columns to float; leave others as-is."""
    numeric_cols = [
        "ts", "id.orig_p", "id.resp_p", "duration",
        "orig_bytes", "resp_bytes", "missed_bytes",
        "orig_pkts", "orig_ip_bytes", "resp_pkts", "resp_ip_bytes",
    ]
    df = df.copy()
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

# ---------------------------------------------------------------------------
# Step 1: Load and prepare raw data
# ---------------------------------------------------------------------------

def load_and_engineer(csv_path: str) -> pd.DataFrame:
    """
    Load the Zeek CSV, cast types, and engineer all behavioural features.
    The 'label' column is preserved for downstream data selection but
    is never passed to the GRU.
    """
    logger.info("Loading CSV: %s", csv_path)
    df = pd.read_csv(csv_path)
    logger.info("Loaded shape: %s", df.shape)

    df = _cast_numeric_columns(df)
    df = engineer_features(df)

    # Confirm ts is numeric and sort chronologically
    if df["ts"].isna().all():
        raise ValueError("All 'ts' values are NaN after casting. Check the CSV.")
    df = df.sort_values("ts").reset_index(drop=True)

    logger.info("Chronologically sorted. ts range: %.3f -- %.3f  (span: %.1f s)",
                df["ts"].min(), df["ts"].max(), df["ts"].max() - df["ts"].min())
    return df


# ---------------------------------------------------------------------------
# Step 2: Aggregate into time-window network-state vectors
# ---------------------------------------------------------------------------

def aggregate_into_windows(df: pd.DataFrame,
                            window_seconds: int = AGGREGATION_WINDOW_SECONDS,
                            label_col: str = "label") -> pd.DataFrame:
    """
    Group rows into fixed time windows and aggregate GRU features.

    Aggregation rules per GRU_FEATURE_SPEC:
      - duration_mean         : mean   (rates / durations averaged)
      - packet_count_sum      : sum    (traffic volume added up)
      - total_bytes_sum       : sum    (traffic volume added up)
      - bytes_per_packet_mean : mean   (per-packet ratio averaged)
      - packet_rate_mean      : mean   (rate averaged)
      - byte_rate_mean        : mean   (rate averaged)
      - sent_received_ratio_mean : mean (ratio averaged; NaN from imputer-free NaN)

    Each window also gets:
      - window_start : timestamp of the window's left edge
      - row_count    : number of rows that fell into the window
      - label_mode   : most-common label in the window (for eval only)

    Forbidden columns (label, ts, etc.) never enter the GRU feature matrix.
    """
    ts_min = df["ts"].min()
    ts_max = df["ts"].max()

    # Assign each row a window index
    df = df.copy()
    df["_win_idx"] = ((df["ts"] - ts_min) / window_seconds).astype(int)

    # Build aggregation dict for GRU features
    agg_dict = {}
    for src_col, agg_fn, out_col in GRU_FEATURE_SPEC:
        agg_dict[src_col] = agg_fn

    # Additional metadata aggregations
    agg_dict["ts"] = "min"
    agg_dict["row_count"] = ("ts", "count")

    # Execute aggregation
    grp = df.groupby("_win_idx")
    records = []
    for win_idx, grp_df in grp:
        row = {"window_idx": win_idx}
        row["window_start"] = grp_df["ts"].min()
        row["row_count"]    = len(grp_df)

        # Aggregate each GRU feature
        for src_col, agg_fn, out_col in GRU_FEATURE_SPEC:
            if src_col not in grp_df.columns:
                row[out_col] = np.nan
                continue
            vals = grp_df[src_col].dropna()
            if len(vals) == 0:
                row[out_col] = np.nan
            elif agg_fn == "mean":
                row[out_col] = float(vals.mean())
            elif agg_fn == "sum":
                row[out_col] = float(vals.sum())
            else:
                row[out_col] = float(vals.agg(agg_fn))

        # Label mode for evaluation purposes only
        if label_col in grp_df.columns:
            row[label_col] = grp_df[label_col].mode().iloc[0] if len(grp_df) > 0 else "Unknown"
        else:
            row[label_col] = "Unknown"

        records.append(row)

    states_df = pd.DataFrame(records)
    states_df = states_df.sort_values("window_start").reset_index(drop=True)

    # Replace inf values with NaN
    for col in GRU_FEATURE_NAMES:
        if col in states_df.columns:
            states_df[col] = states_df[col].replace([np.inf, -np.inf], np.nan)

    logger.info("Aggregated %d rows into %d network-state windows (window=%ds)",
                len(df), len(states_df), window_seconds)
    logger.info("Window feature NaN counts:\n%s",
                states_df[GRU_FEATURE_NAMES].isna().sum().to_string())

    return states_df


# ---------------------------------------------------------------------------
# Step 3: Construct sliding-window sequences
# ---------------------------------------------------------------------------

def build_sequences(states_df: pd.DataFrame,
                    seq_len: int = SEQUENCE_LENGTH) -> tuple:
    """
    Build (X, y, meta) from the ordered network-state DataFrame.

    X.shape = (N, seq_len, N_GRU_FEATURES)
    y.shape = (N, N_GRU_FEATURES)
    meta    = list of dicts with 'window_start' for the TARGET step

    Missing values in feature columns are filled with column median
    (computed only from this states_df, not the full dataset) so the
    GRU always receives finite inputs.

    Forbidden features are verified absent before sequence construction.
    """
    # Safety: assert forbidden features are not in the feature matrix
    forbidden_present = _FORBIDDEN_FEATURES.intersection(set(GRU_FEATURE_NAMES))
    if forbidden_present:
        raise ValueError(
            f"CRITICAL: forbidden features in GRU_FEATURE_NAMES: {forbidden_present}"
        )

    feature_df = states_df[GRU_FEATURE_NAMES].copy()

    # Fill NaN with column median (computed within this split only)
    for col in GRU_FEATURE_NAMES:
        median = feature_df[col].median()
        if np.isnan(median):
            median = 0.0
        feature_df[col] = feature_df[col].fillna(median)

    # Verify no remaining NaN or inf
    if feature_df.isna().any().any():
        raise ValueError("NaN values remain in feature matrix after median fill.")
    if np.isinf(feature_df.values).any():
        raise ValueError("Inf values remain in feature matrix.")

    values = feature_df.values  # shape: (n_windows, N_GRU_FEATURES)

    n_windows = len(values)
    if n_windows < seq_len + 1:
        raise ValueError(
            f"Insufficient windows ({n_windows}) to build sequences of "
            f"length {seq_len}+1. Need at least {seq_len + 1} windows."
        )

    X_list, y_list, meta_list = [], [], []
    for i in range(n_windows - seq_len):
        X_list.append(values[i : i + seq_len])
        y_list.append(values[i + seq_len])
        meta_list.append({
            "window_start": states_df["window_start"].iloc[i + seq_len],
            "label":        states_df["label"].iloc[i + seq_len]
                            if "label" in states_df.columns else "Unknown",
        })

    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.stack(y_list, axis=0).astype(np.float32)

    logger.info("Sequences built: X=%s  y=%s", X.shape, y.shape)
    return X, y, meta_list


# ---------------------------------------------------------------------------
# Step 4: Chronological train / validation split
# ---------------------------------------------------------------------------

def temporal_train_val_split(states_df: pd.DataFrame,
                              train_frac: float = 0.80) -> tuple:
    """
    Split the ordered network-state DataFrame chronologically.

    The first train_frac of WINDOWS become the training set.
    The remaining windows become the validation set.

    NO random shuffling. The boundary is a clean temporal cut.

    Returns: (df_train, df_val, split_timestamp)
    """
    n = len(states_df)
    split_idx = int(n * train_frac)
    if split_idx < SEQUENCE_LENGTH + 1:
        raise ValueError(
            f"Too few training windows ({split_idx}) for seq_len={SEQUENCE_LENGTH}."
        )
    if n - split_idx < SEQUENCE_LENGTH + 1:
        raise ValueError(
            f"Too few validation windows ({n - split_idx}) for seq_len={SEQUENCE_LENGTH}."
        )

    df_train = states_df.iloc[:split_idx].reset_index(drop=True)
    df_val   = states_df.iloc[split_idx:].reset_index(drop=True)
    split_ts = float(states_df["window_start"].iloc[split_idx])

    logger.info(
        "Temporal split: %d train windows | %d val windows | boundary ts=%.3f",
        len(df_train), len(df_val), split_ts
    )
    return df_train, df_val, split_ts


# ---------------------------------------------------------------------------
# Full preprocessing pipeline (entry point for training)
# ---------------------------------------------------------------------------

def run_full_preprocessing(csv_path: str = PROCESSED_CSV_PATH,
                            window_seconds: int = AGGREGATION_WINDOW_SECONDS,
                            seq_len: int = SEQUENCE_LENGTH,
                            train_frac: float = 0.80) -> dict:
    """
    Full GRU preprocessing pipeline:
      1. Load raw CSV and engineer features
      2. Filter to normal traffic only (Background + Benign)
      3. Aggregate into time-window network states
      4. Chronological train/val split
      5. Build sequences for each split

    Returns a dict with:
      "X_train", "y_train", "meta_train"
      "X_val",   "y_val",   "meta_val"
      "df_states_normal"   : all normal states (pre-split)
      "df_states_all"      : all states (for malicious evaluation)
      "split_ts"           : exact chronological split boundary
      "feature_names"      : GRU_FEATURE_NAMES
      "n_features"         : N_GRU_FEATURES
      "window_seconds"     : window_seconds
      "seq_len"            : seq_len
    """
    # Step 1: load & engineer
    df = load_and_engineer(csv_path)
    ts_min = df["ts"].min()
    ts_max = df["ts"].max()
    logger.info("Dataset time range: %.3f -- %.3f  (span: %.1f s)", ts_min, ts_max, ts_max - ts_min)

    # Step 2: filter normal traffic
    normal_labels = ["Background", "Benign"]
    df_normal = df[df["label"].isin(normal_labels)].copy()
    df_mal    = df[df["label"] == "Malicious"].copy()
    logger.info("Normal rows: %d  |  Malicious rows: %d", len(df_normal), len(df_mal))

    # Step 3a: aggregate normal data
    states_normal = aggregate_into_windows(df_normal, window_seconds)
    logger.info("Normal network states: %d", len(states_normal))

    # Step 3b: aggregate ALL data (for malicious evaluation later)
    states_all = aggregate_into_windows(df, window_seconds)
    logger.info("All-data network states: %d", len(states_all))

    # Step 4: chronological split on normal states only
    df_train_states, df_val_states, split_ts = temporal_train_val_split(
        states_normal, train_frac
    )

    # Step 5: build sequences
    logger.info("Building training sequences ...")
    X_train, y_train, meta_train = build_sequences(df_train_states, seq_len)

    logger.info("Building validation sequences ...")
    X_val, y_val, meta_val = build_sequences(df_val_states, seq_len)

    logger.info("=" * 55)
    logger.info("GRU Preprocessing Summary")
    logger.info("=" * 55)
    logger.info("  Aggregation window : %d s", window_seconds)
    logger.info("  Sequence length    : %d", seq_len)
    logger.info("  Normal states      : %d", len(states_normal))
    logger.info("  Train windows      : %d  (split boundary: %.3f)", len(df_train_states), split_ts)
    logger.info("  Val   windows      : %d", len(df_val_states))
    logger.info("  X_train shape      : %s", X_train.shape)
    logger.info("  y_train shape      : %s", y_train.shape)
    logger.info("  X_val shape        : %s", X_val.shape)
    logger.info("  y_val shape        : %s", y_val.shape)
    logger.info("  GRU feature names  : %s", GRU_FEATURE_NAMES)
    logger.info("  n_features         : %d", N_GRU_FEATURES)
    logger.info("  Forbidden features in X/y: %s", _FORBIDDEN_FEATURES)
    logger.info("=" * 55)

    return {
        "X_train":           X_train,
        "y_train":           y_train,
        "meta_train":        meta_train,
        "X_val":             X_val,
        "y_val":             y_val,
        "meta_val":          meta_val,
        "df_states_normal":  states_normal,
        "df_states_all":     states_all,
        "split_ts":          split_ts,
        "feature_names":     list(GRU_FEATURE_NAMES),
        "n_features":        N_GRU_FEATURES,
        "window_seconds":    window_seconds,
        "seq_len":           seq_len,
    }


# ---------------------------------------------------------------------------
# Malicious evaluation data (separate from training)
# ---------------------------------------------------------------------------

def get_malicious_sequences(df_states_all: pd.DataFrame,
                             split_ts: float,
                             seq_len: int = SEQUENCE_LENGTH) -> tuple:
    """
    Extract sequences from malicious-dominated windows AFTER the split boundary.
    Uses ALL windows (not normal-only) to capture malicious periods.

    Returns (X_mal, y_mal, meta_mal) or (None, None, None) if insufficient.
    """
    # Use full-data states where malicious label predominates
    df_mal_states = df_states_all[
        df_states_all["label"] == "Malicious"
    ].copy().reset_index(drop=True)

    if len(df_mal_states) < seq_len + 1:
        logger.warning(
            "Insufficient malicious states (%d) for seq_len=%d. Skipping.",
            len(df_mal_states), seq_len
        )
        return None, None, None

    try:
        X_mal, y_mal, meta_mal = build_sequences(df_mal_states, seq_len)
        logger.info("Malicious sequences: X=%s  y=%s", X_mal.shape, y_mal.shape)
        return X_mal, y_mal, meta_mal
    except ValueError as e:
        logger.warning("Could not build malicious sequences: %s", e)
        return None, None, None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    result = run_full_preprocessing()

    print("\n" + "=" * 60)
    print("  GRU Preprocessing Results")
    print("=" * 60)
    print(f"  Dataset time range    : {result['df_states_normal']['window_start'].min():.3f}"
          f"  --  {result['df_states_all']['window_start'].max():.3f}")
    print(f"  Aggregation window    : {result['window_seconds']} s")
    print(f"  Sequence length       : {result['seq_len']}")
    print(f"  Normal states total   : {len(result['df_states_normal'])}")
    print(f"  X_train shape         : {result['X_train'].shape}")
    print(f"  y_train shape         : {result['y_train'].shape}")
    print(f"  X_val shape           : {result['X_val'].shape}")
    print(f"  y_val shape           : {result['y_val'].shape}")
    print(f"  GRU feature names ({result['n_features']}):")
    for i, n in enumerate(result["feature_names"], 1):
        print(f"    {i}. {n}")
    print(f"  Split boundary ts     : {result['split_ts']:.3f}")
    print("  Leakage check:")
    print(f"    label in features   : {'label' in result['feature_names']}")
    print(f"    ts in features      : {'ts' in result['feature_names']}")
    print(f"    detailedlabel in f  : {'detailedlabel' in result['feature_names']}")
