"""
TRINETRA -- gru_inference.py
=============================
GRU Component, Step 3: Inference -- trajectory deviation for new data.

Interface
---------
    from gru_inference import predict_trajectory, load_gru_artefacts

    # Load saved artefacts once
    model, scaler, config = load_gru_artefacts()

    # Score new data
    result = predict_trajectory(network_data_df, model, scaler, config)
    # result is a DataFrame with columns:
    #   window_start, trajectory_deviation, predicted_state, actual_state

    # Or convenience wrapper:
    result = run_inference_on_csv("data/processed/labeled-conn.csv")

Design contract
---------------
* Loads gru_trajectory.pt, gru_scaler.joblib, gru_config.joblib
* Does NOT refit the scaler
* Does NOT retrain the GRU
* Validates incoming data against saved config (feature names, count)
* Raises clear ValueError if required features are missing
* Extra columns in the DataFrame are silently ignored
* timestamp (ts) is used for ordering only, never as a GRU feature
* label / detailedlabel are kept in output for context, never in X
* Trajectory deviation = ||actual_state - predicted_state||_2 (Euclidean)

All imports are at the top of the file.
"""

import os
import sys
import logging

import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gru_preprocessing import (
    _cast_numeric_columns,
    engineer_features,
    aggregate_into_windows,
    build_sequences,
    AGGREGATION_WINDOW_SECONDS,
    SEQUENCE_LENGTH,
    GRU_FEATURE_NAMES,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRU_MODEL_PATH  = os.path.join(PROJECT_ROOT, "models", "gru_trajectory.pt")
GRU_SCALER_PATH = os.path.join(PROJECT_ROOT, "models", "gru_scaler.joblib")
GRU_CONFIG_PATH = os.path.join(PROJECT_ROOT, "models", "gru_config.joblib")
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

# Forbidden features that must never enter the GRU
_FORBIDDEN = {"label", "detailedlabel", "ts", "anomaly_score",
              "predicted_anomaly", "anomaly_score_raw", "anomaly_score_negated"}


# ---------------------------------------------------------------------------
# GRU Model Definition (must match train_gru.py exactly)
# ---------------------------------------------------------------------------

class TrajectoryGRU(nn.Module):
    """
    Single-layer GRU for next-state trajectory prediction.
    Architecture must match the saved model exactly.
    """
    def __init__(self, input_size: int, hidden_size: int = 32):
        super().__init__()
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.gru    = nn.GRU(input_size=input_size,
                             hidden_size=hidden_size,
                             num_layers=1,
                             batch_first=True)
        self.linear = nn.Linear(hidden_size, input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h_n = self.gru(x)
        h_last = h_n.squeeze(0)
        return self.linear(h_last)


# ---------------------------------------------------------------------------
# Artefact loading
# ---------------------------------------------------------------------------

def load_gru_artefacts(model_path: str = GRU_MODEL_PATH,
                        scaler_path: str = GRU_SCALER_PATH,
                        config_path: str = GRU_CONFIG_PATH) -> tuple:
    """
    Load the three saved GRU artefacts from disk.

    Returns: (model, scaler, config)
      model  : TrajectoryGRU with loaded weights (eval mode)
      scaler : fitted StandardScaler
      config : dict with feature_names, sequence_length, etc.

    Raises FileNotFoundError if any artefact is missing.
    """
    for path in (model_path, scaler_path, config_path):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Required GRU artefact not found: {path}\n"
                "Run train_gru.py first."
            )

    config = joblib.load(config_path)
    n_features  = config["n_features"]
    hidden_size = config["hidden_size"]

    model = TrajectoryGRU(input_size=n_features, hidden_size=hidden_size)
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    scaler = joblib.load(scaler_path)

    logger.info("GRU model loaded   : %s", model_path)
    logger.info("GRU scaler loaded  : %s", scaler_path)
    logger.info("GRU config loaded  : n_features=%d, seq_len=%d, window=%ds",
                n_features, config["sequence_length"], config["aggregation_window"])

    return model, scaler, config


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def validate_input_schema(states_df: pd.DataFrame, config: dict) -> None:
    """
    Verify the aggregated states DataFrame contains all required GRU features
    in the correct order, and that no forbidden features are present.

    Raises ValueError if required features are missing.
    Extra columns are silently ignored.
    """
    required = config["feature_names"]

    # Forbidden features must not have entered the schema
    forbidden_in_req = _FORBIDDEN.intersection(set(required))
    if forbidden_in_req:
        raise ValueError(
            f"CRITICAL: saved config contains forbidden features: {forbidden_in_req}"
        )

    missing = [f for f in required if f not in states_df.columns]
    if missing:
        raise ValueError(
            f"Input states DataFrame missing {len(missing)} required GRU feature(s). "
            f"Missing: {missing}\n"
            f"Available: {sorted(states_df.columns.tolist())}"
        )

    logger.info(
        "Input schema validated: all %d required features present.", len(required)
    )


# ---------------------------------------------------------------------------
# Core inference
# ---------------------------------------------------------------------------

def _scale_sequences(X: np.ndarray, scaler) -> np.ndarray:
    """Apply the fitted scaler to X without refitting."""
    n, seq_len, n_f = X.shape
    return scaler.transform(X.reshape(-1, n_f)).reshape(n, seq_len, n_f).astype(np.float32)


def _predict_batch(model: TrajectoryGRU, X_sc: np.ndarray) -> np.ndarray:
    """Run the GRU on scaled input and return scaled predictions."""
    model.eval()
    with torch.no_grad():
        X_t    = torch.tensor(X_sc, dtype=torch.float32)
        y_pred = model(X_t).numpy()
    return y_pred


def predict_trajectory(df: pd.DataFrame,
                        model: TrajectoryGRU,
                        scaler,
                        config: dict) -> pd.DataFrame:
    """
    Run GRU trajectory prediction on a DataFrame of Zeek connection records.

    Pipeline:
      1. Cast numeric columns
      2. Engineer behavioural features
      3. Sort chronologically
      4. Aggregate into time windows (using config["aggregation_window"])
      5. Build sliding-window sequences (using config["sequence_length"])
      6. Scale with saved scaler (no refit)
      7. Run GRU -> predicted next state
      8. Inverse-scale actual and predicted states
      9. Compute trajectory deviation = ||actual - predicted||_2

    Returns a DataFrame with columns:
      window_start          : timestamp of the target window
      trajectory_deviation  : Euclidean distance (float)
      predicted_state       : list of predicted feature values (unscaled)
      actual_state          : list of actual feature values (unscaled)
      label                 : window label if available (context only, not a GRU input)

    Raises ValueError if required features are missing or dimensions mismatch.
    """
    seq_len     = config["sequence_length"]
    window_secs = config["aggregation_window"]
    n_features  = config["n_features"]
    feat_names  = config["feature_names"]

    # Step 1-2: cast + engineer
    df = _cast_numeric_columns(df.copy())
    df = engineer_features(df)

    if "ts" not in df.columns or df["ts"].isna().all():
        raise ValueError("Column 'ts' is required for temporal ordering but is missing or all-NaN.")
    df = df.sort_values("ts").reset_index(drop=True)

    # Step 3: aggregate into windows
    states_df = aggregate_into_windows(df, window_seconds=window_secs)
    logger.info("Aggregated to %d network-state windows.", len(states_df))

    # Step 4: validate schema
    validate_input_schema(states_df, config)

    # Step 5: build sequences
    if len(states_df) < seq_len + 1:
        raise ValueError(
            f"Insufficient windows ({len(states_df)}) for seq_len={seq_len}. "
            f"Need at least {seq_len + 1}."
        )
    X_raw, y_raw, meta = build_sequences(states_df, seq_len)

    logger.info("Sequences: X=%s  y=%s", X_raw.shape, y_raw.shape)

    # Verify dimensions match saved config
    if X_raw.shape[2] != n_features:
        raise ValueError(
            f"Feature dimension mismatch: input has {X_raw.shape[2]} features "
            f"but config expects {n_features}."
        )

    # Step 6: scale
    X_sc = _scale_sequences(X_raw, scaler)
    y_sc = scaler.transform(y_raw)

    # Step 7: predict
    y_pred_sc = _predict_batch(model, X_sc)

    # Step 8: inverse-scale
    y_actual_unscaled = scaler.inverse_transform(y_raw)
    y_pred_unscaled   = scaler.inverse_transform(y_pred_sc)

    # Step 9: trajectory deviation
    diff       = y_actual_unscaled - y_pred_unscaled
    deviations = np.sqrt((diff ** 2).sum(axis=1))

    # Build result DataFrame
    rows = []
    for i, m in enumerate(meta):
        rows.append({
            "window_start":        m["window_start"],
            "label":               m.get("label", "Unknown"),
            "trajectory_deviation": float(deviations[i]),
            "predicted_state":     y_pred_unscaled[i].tolist(),
            "actual_state":        y_actual_unscaled[i].tolist(),
        })
    result_df = pd.DataFrame(rows).sort_values("window_start").reset_index(drop=True)

    n_total = len(result_df)
    logger.info("Inference complete: %d sequences scored.", n_total)
    logger.info("Trajectory deviation: mean=%.4f  median=%.4f  max=%.4f",
                deviations.mean(), float(np.median(deviations)), deviations.max())

    return result_df


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------

def run_inference_on_csv(csv_path: str,
                          model_path: str = GRU_MODEL_PATH,
                          scaler_path: str = GRU_SCALER_PATH,
                          config_path: str = GRU_CONFIG_PATH) -> pd.DataFrame:
    """
    Load artefacts and run inference on a CSV file.
    Returns the trajectory result DataFrame.
    """
    logger.info("Loading CSV for GRU inference: %s", csv_path)
    df = pd.read_csv(csv_path)
    model, scaler, config = load_gru_artefacts(model_path, scaler_path, config_path)
    return predict_trajectory(df, model, scaler, config)


def run_inference_on_dataframe(df: pd.DataFrame,
                                model_path: str = GRU_MODEL_PATH,
                                scaler_path: str = GRU_SCALER_PATH,
                                config_path: str = GRU_CONFIG_PATH) -> pd.DataFrame:
    """
    Load artefacts and run inference on an existing DataFrame.
    Useful for the integration layer (Isolation Forest + GRU combined pipeline).
    """
    model, scaler, config = load_gru_artefacts(model_path, scaler_path, config_path)
    return predict_trajectory(df, model, scaler, config)


# ---------------------------------------------------------------------------
# Integration-style test (standalone, no retraining)
# ---------------------------------------------------------------------------

def integration_test(csv_path: str = PROCESSED_CSV_PATH) -> None:
    """
    Demonstrate the full inference pipeline without retraining.

    This proves that another TRINETRA module can:
      1. Load the saved GRU artefacts
      2. Pass canonical network data
      3. Receive structured trajectory_deviation output

    without any knowledge of the training implementation.
    """
    logger.info("=" * 60)
    logger.info("GRU Integration Test (no retraining)")
    logger.info("=" * 60)

    # Load artefacts
    model, scaler, config = load_gru_artefacts()

    logger.info("Config feature_names : %s", config["feature_names"])
    logger.info("Config seq_len       : %d", config["sequence_length"])
    logger.info("Config window        : %d s", config["aggregation_window"])

    # Load canonical data
    df = pd.read_csv(csv_path)
    logger.info("Loaded %d rows from %s", len(df), csv_path)

    # Run inference
    result = predict_trajectory(df, model, scaler, config)

    logger.info("Result shape: %s", result.shape)
    logger.info("Result columns: %s", result.columns.tolist())

    # Print sample structured output
    print("\n" + "=" * 65)
    print("  GRU Integration Test -- Sample Output")
    print("=" * 65)
    print(result[["window_start", "label", "trajectory_deviation"]].head(10).to_string(index=False))
    print(f"\n  Total sequences scored : {len(result)}")
    print(f"  Mean deviation         : {result['trajectory_deviation'].mean():.4f}")
    print(f"  Median deviation       : {result['trajectory_deviation'].median():.4f}")
    print(f"  95th pct deviation     : {result['trajectory_deviation'].quantile(0.95):.4f}")

    # Demonstrate machine-readable access
    print("\n  Example structured result (first sequence):")
    row = result.iloc[0]
    example = {
        "window_start":        row["window_start"],
        "trajectory_deviation": row["trajectory_deviation"],
        "predicted_state":     row["predicted_state"],
        "actual_state":        row["actual_state"],
    }
    for k, v in example.items():
        if isinstance(v, list):
            print(f"    {k}: {[round(x, 4) for x in v]}")
        else:
            print(f"    {k}: {v}")

    print("\n  Integration test PASSED: GRU artefacts loaded and scored without retraining.")
    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="TRINETRA GRU -- trajectory deviation inference"
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        default=PROCESSED_CSV_PATH,
        help="Path to Zeek CSV file (default: labeled-conn.csv).",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Optional path to write scored results CSV.",
    )
    parser.add_argument(
        "--integration-test",
        action="store_true",
        help="Run the integration test demonstrating standalone inference.",
    )
    args = parser.parse_args()

    if args.integration_test:
        result = integration_test(args.input_path)
    else:
        result = run_inference_on_csv(args.input_path)

        n = len(result)
        print("\n" + "=" * 60)
        print("  TRINETRA GRU -- Inference Summary")
        print("=" * 60)
        print(f"  Sequences scored     : {n}")
        print(f"  Mean deviation       : {result['trajectory_deviation'].mean():.4f}")
        print(f"  Median deviation     : {result['trajectory_deviation'].median():.4f}")
        print(f"  95th pct deviation   : {result['trajectory_deviation'].quantile(0.95):.4f}")
        print(f"  Max deviation        : {result['trajectory_deviation'].max():.4f}")
        print("\n--- Top 10 highest trajectory deviations ---")
        top = result.sort_values("trajectory_deviation", ascending=False).head(10)
        print(top[["window_start", "label", "trajectory_deviation"]].to_string(index=False))

    if args.output:
        result.to_csv(args.output, index=False)
        logger.info("Results written to: %s", args.output)
