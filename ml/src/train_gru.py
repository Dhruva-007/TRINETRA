"""
TRINETRA -- train_gru.py
=========================
GRU Component, Step 2: Train the GRU next-state prediction model.

Pipeline:
  raw Zeek CSV
      |
      v
  gru_preprocessing.py  (temporal aggregation + sequence construction)
      |
      v
  StandardScaler  (fit on training data ONLY)
      |
      v
  GRU(input_size=N_FEATURES, hidden_size=32)
      |
      v
  Linear(32, N_FEATURES)
      |
      v
  MSELoss (next-state prediction)
      |
      v
  saved: gru_trajectory.pt  gru_scaler.joblib  gru_config.joblib

GRU learns: X(t-4..t) --> predicted X(t+1)
Target    : actual X(t+1)
Deviation : ||actual - predicted||_2

Labels are NEVER used as GRU inputs.
Scaler is fitted ONLY on training data.

All imports are at the top of the file.
"""

import os
import sys
import json
import logging
import random

import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gru_preprocessing import (
    run_full_preprocessing,
    get_malicious_sequences,
    GRU_FEATURE_NAMES,
    N_GRU_FEATURES,
    AGGREGATION_WINDOW_SECONDS,
    SEQUENCE_LENGTH,
    PROCESSED_CSV_PATH,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRU_MODEL_PATH  = os.path.join(PROJECT_ROOT, "models", "gru_trajectory.pt")
GRU_SCALER_PATH = os.path.join(PROJECT_ROOT, "models", "gru_scaler.joblib")
GRU_CONFIG_PATH = os.path.join(PROJECT_ROOT, "models", "gru_config.joblib")
HISTORY_PATH    = os.path.join(PROJECT_ROOT, "results", "gru_training_history.json")
METRICS_PATH    = os.path.join(PROJECT_ROOT, "results", "gru_metrics.json")
SCORES_CSV_PATH = os.path.join(PROJECT_ROOT, "results", "gru_trajectory_scores.csv")

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
# Reproducibility
# ---------------------------------------------------------------------------

SEED = 42

def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

HIDDEN_SIZE    = 32
EPOCHS         = 50
BATCH_SIZE     = 32
LEARNING_RATE  = 0.001
EARLY_STOP_PATIENCE = 10   # stop if val loss does not improve for N epochs

# ---------------------------------------------------------------------------
# GRU Model Definition
# ---------------------------------------------------------------------------

class TrajectoryGRU(nn.Module):
    """
    Single-layer GRU for next-state trajectory prediction.

    Architecture:
        Input  : (batch, seq_len, n_features)
        GRU    : hidden_size=32, batch_first=True
        Linear : 32 -> n_features
        Output : (batch, n_features)  -- predicted next state

    Intentionally small and explainable.
    No bidirectional, no attention, no Transformer layers.
    """

    def __init__(self, input_size: int, hidden_size: int = HIDDEN_SIZE):
        super().__init__()
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.gru    = nn.GRU(input_size=input_size,
                             hidden_size=hidden_size,
                             num_layers=1,
                             batch_first=True)
        self.linear = nn.Linear(hidden_size, input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x   : (batch, seq_len, input_size)
        out : (batch, input_size)  -- predicted next state
        """
        _, h_n = self.gru(x)       # h_n: (1, batch, hidden_size)
        h_last = h_n.squeeze(0)    # (batch, hidden_size)
        return self.linear(h_last) # (batch, input_size)


# ---------------------------------------------------------------------------
# Scaler helpers
# ---------------------------------------------------------------------------

def fit_scaler_on_train(X_train: np.ndarray) -> StandardScaler:
    """
    Fit a StandardScaler on training data ONLY (requirement: no leakage).

    X_train.shape = (N, seq_len, n_features)
    We reshape to (N * seq_len, n_features) to compute column-wise stats.
    """
    n, seq_len, n_f = X_train.shape
    X_flat = X_train.reshape(-1, n_f)
    scaler = StandardScaler()
    scaler.fit(X_flat)
    logger.info("Scaler fitted on %d training state vectors.", n * seq_len)
    return scaler


def apply_scaler(X: np.ndarray, y: np.ndarray,
                 scaler: StandardScaler) -> tuple:
    """
    Apply a FITTED scaler to X and y WITHOUT refitting.

    X.shape = (N, seq_len, n_features)
    y.shape = (N, n_features)
    """
    n, seq_len, n_f = X.shape
    X_scaled = scaler.transform(X.reshape(-1, n_f)).reshape(n, seq_len, n_f)
    y_scaled = scaler.transform(y)
    return X_scaled.astype(np.float32), y_scaled.astype(np.float32)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_epoch(model: nn.Module,
                loader: DataLoader,
                optimizer: torch.optim.Optimizer,
                criterion: nn.Module,
                device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss   = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(X_batch)
    return total_loss / len(loader.dataset)


def val_epoch(model: nn.Module,
              loader: DataLoader,
              criterion: nn.Module,
              device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            y_pred  = model(X_batch)
            loss    = criterion(y_pred, y_batch)
            total_loss += loss.item() * len(X_batch)
    return total_loss / len(loader.dataset)


def train_gru_model(X_train_sc: np.ndarray,
                    y_train_sc: np.ndarray,
                    X_val_sc:   np.ndarray,
                    y_val_sc:   np.ndarray,
                    n_features: int,
                    hidden_size: int = HIDDEN_SIZE,
                    epochs: int = EPOCHS,
                    batch_size: int = BATCH_SIZE,
                    lr: float = LEARNING_RATE) -> tuple:
    """
    Train the GRU model with early stopping on validation MSE.
    Returns (best_model, history_dict).
    """
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training device: %s", device)

    model     = TrajectoryGRU(input_size=n_features, hidden_size=hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    X_tr = torch.tensor(X_train_sc, dtype=torch.float32)
    y_tr = torch.tensor(y_train_sc, dtype=torch.float32)
    X_vl = torch.tensor(X_val_sc,   dtype=torch.float32)
    y_vl = torch.tensor(y_val_sc,   dtype=torch.float32)

    train_ds = TensorDataset(X_tr, y_tr)
    val_ds   = TensorDataset(X_vl, y_vl)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    history = {"train_loss": [], "val_loss": []}
    best_val_loss  = float("inf")
    best_state     = None
    patience_count = 0

    logger.info("Starting training: epochs=%d  batch=%d  lr=%s  hidden=%d",
                epochs, batch_size, lr, hidden_size)
    logger.info("%-8s  %-14s  %-14s", "Epoch", "Train Loss", "Val Loss")
    logger.info("-" * 40)

    for epoch in range(1, epochs + 1):
        tr_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        vl_loss = val_epoch(model, val_loader, criterion, device)

        history["train_loss"].append(round(tr_loss, 6))
        history["val_loss"].append(round(vl_loss, 6))

        improved = vl_loss < best_val_loss
        if improved:
            best_val_loss  = vl_loss
            best_state     = {k: v.clone().cpu() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1

        marker = " *" if improved else ""
        logger.info("Epoch %3d | train_loss=%.6f | val_loss=%.6f%s",
                    epoch, tr_loss, vl_loss, marker)

        if patience_count >= EARLY_STOP_PATIENCE:
            logger.info("Early stopping at epoch %d (patience=%d).",
                        epoch, EARLY_STOP_PATIENCE)
            break

    # Restore best weights
    model.load_state_dict(best_state)
    logger.info("Best val_loss: %.6f", best_val_loss)
    return model, history


# ---------------------------------------------------------------------------
# Trajectory deviation
# ---------------------------------------------------------------------------

def compute_trajectory_deviation(model: nn.Module,
                                  X_sc: np.ndarray,
                                  y_sc: np.ndarray,
                                  scaler: StandardScaler,
                                  device: torch.device) -> np.ndarray:
    """
    Compute per-sequence trajectory deviation in the ORIGINAL (unscaled) space.

    Steps:
      1. Run model on scaled X -> scaled predicted next state
      2. Inverse-transform predicted and actual to original scale
      3. Euclidean distance ||actual - predicted||_2

    Returns: deviations array of shape (N,)
    """
    model.eval()
    X_t = torch.tensor(X_sc, dtype=torch.float32).to(device)
    with torch.no_grad():
        y_pred_sc = model(X_t).cpu().numpy()

    # Inverse-transform to original scale
    y_actual_orig = scaler.inverse_transform(y_sc)
    y_pred_orig   = scaler.inverse_transform(y_pred_sc)

    # Euclidean distance per sequence
    diff = y_actual_orig - y_pred_orig
    deviations = np.sqrt((diff ** 2).sum(axis=1))
    return deviations, y_pred_orig, y_actual_orig


# ---------------------------------------------------------------------------
# Save artefacts
# ---------------------------------------------------------------------------

def save_artefacts(model: nn.Module,
                   scaler: StandardScaler,
                   config: dict,
                   history: dict,
                   metrics: dict,
                   scores_df: pd.DataFrame):
    """Save all GRU artefacts to models/ and results/."""
    os.makedirs(os.path.dirname(GRU_MODEL_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)

    torch.save(model.state_dict(), GRU_MODEL_PATH)
    logger.info("GRU model saved     : %s", GRU_MODEL_PATH)

    joblib.dump(scaler, GRU_SCALER_PATH)
    logger.info("GRU scaler saved    : %s", GRU_SCALER_PATH)

    joblib.dump(config, GRU_CONFIG_PATH)
    logger.info("GRU config saved    : %s", GRU_CONFIG_PATH)

    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)
    logger.info("Training history    : %s", HISTORY_PATH)

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("GRU metrics saved   : %s", METRICS_PATH)

    scores_df.to_csv(SCORES_CSV_PATH, index=False)
    logger.info("Trajectory scores   : %s", SCORES_CSV_PATH)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("=" * 60)
    logger.info("TRINETRA -- GRU Trajectory Prediction Training")
    logger.info("=" * 60)

    # -----------------------------------------------------------------------
    # Step 1: Preprocessing
    # -----------------------------------------------------------------------
    prep = run_full_preprocessing(PROCESSED_CSV_PATH,
                                   window_seconds=AGGREGATION_WINDOW_SECONDS,
                                   seq_len=SEQUENCE_LENGTH)

    X_train_raw = prep["X_train"]
    y_train_raw = prep["y_train"]
    X_val_raw   = prep["X_val"]
    y_val_raw   = prep["y_val"]
    meta_train  = prep["meta_train"]
    meta_val    = prep["meta_val"]

    logger.info("X_train: %s  y_train: %s", X_train_raw.shape, y_train_raw.shape)
    logger.info("X_val  : %s  y_val  : %s", X_val_raw.shape,   y_val_raw.shape)

    # Leakage checks
    assert "label"        not in prep["feature_names"], "LEAKAGE: label in features"
    assert "detailedlabel" not in prep["feature_names"], "LEAKAGE: detailedlabel in features"
    assert "ts"           not in prep["feature_names"], "LEAKAGE: ts in features"
    logger.info("Leakage check PASSED: no forbidden features in X/y.")

    # -----------------------------------------------------------------------
    # Step 2: Fit scaler on training data ONLY
    # -----------------------------------------------------------------------
    scaler = fit_scaler_on_train(X_train_raw)

    X_train_sc, y_train_sc = apply_scaler(X_train_raw, y_train_raw, scaler)
    X_val_sc,   y_val_sc   = apply_scaler(X_val_raw,   y_val_raw,   scaler)

    logger.info("Scaler fitted on training data only (no leakage from val/test).")

    # -----------------------------------------------------------------------
    # Step 3: Train GRU
    # -----------------------------------------------------------------------
    model, history = train_gru_model(
        X_train_sc, y_train_sc,
        X_val_sc,   y_val_sc,
        n_features   = prep["n_features"],
        hidden_size  = HIDDEN_SIZE,
        epochs       = EPOCHS,
        batch_size   = BATCH_SIZE,
        lr           = LEARNING_RATE,
    )

    # -----------------------------------------------------------------------
    # Step 4: Compute trajectory deviation on validation data
    # -----------------------------------------------------------------------
    val_devs, val_pred, val_actual = compute_trajectory_deviation(
        model, X_val_sc, y_val_sc, scaler, device
    )
    p95_normal = float(np.percentile(val_devs, 95))

    logger.info("Normal validation trajectory deviation:")
    logger.info("  mean   : %.4f", val_devs.mean())
    logger.info("  median : %.4f", np.median(val_devs))
    logger.info("  std    : %.4f", val_devs.std())
    logger.info("  95pct  : %.4f  (dataset-specific threshold)", p95_normal)
    logger.info("  max    : %.4f", val_devs.max())

    # -----------------------------------------------------------------------
    # Step 5: Evaluate on malicious sequences (if available)
    # -----------------------------------------------------------------------
    X_mal_raw, y_mal_raw, meta_mal = get_malicious_sequences(
        prep["df_states_all"], prep["split_ts"], SEQUENCE_LENGTH
    )

    mal_stats = {}
    mal_deviations = None
    if X_mal_raw is not None:
        X_mal_sc, y_mal_sc = apply_scaler(X_mal_raw, y_mal_raw, scaler)
        mal_devs, mal_pred, mal_actual = compute_trajectory_deviation(
            model, X_mal_sc, y_mal_sc, scaler, device
        )
        mal_stats = {
            "n_sequences":        int(len(mal_devs)),
            "mean_deviation":     round(float(mal_devs.mean()), 4),
            "median_deviation":   round(float(np.median(mal_devs)), 4),
            "std_deviation":      round(float(mal_devs.std()), 4),
            "p95_deviation":      round(float(np.percentile(mal_devs, 95)), 4),
            "max_deviation":      round(float(mal_devs.max()), 4),
        }
        mal_deviations = mal_devs
        logger.info("Malicious trajectory deviation:")
        logger.info("  n_sequences: %d", mal_stats["n_sequences"])
        logger.info("  mean   : %.4f", mal_stats["mean_deviation"])
        logger.info("  median : %.4f", mal_stats["median_deviation"])
        logger.info("  std    : %.4f", mal_stats["std_deviation"])
        logger.info("  95pct  : %.4f", mal_stats["p95_deviation"])

    # -----------------------------------------------------------------------
    # Step 6: Build trajectory scores DataFrame
    # -----------------------------------------------------------------------
    records = []
    for i, m in enumerate(meta_val):
        records.append({
            "window_start":        m["window_start"],
            "label":               m.get("label", "Normal"),
            "split":               "val_normal",
            "trajectory_deviation": round(float(val_devs[i]), 6),
        })
    if mal_deviations is not None:
        for i, m in enumerate(meta_mal):
            records.append({
                "window_start":        m["window_start"],
                "label":               m.get("label", "Malicious"),
                "split":               "malicious",
                "trajectory_deviation": round(float(mal_deviations[i]), 6),
            })
    scores_df = pd.DataFrame(records).sort_values("window_start").reset_index(drop=True)

    # -----------------------------------------------------------------------
    # Step 7: Build config and metrics
    # -----------------------------------------------------------------------
    config = {
        "feature_names":       list(prep["feature_names"]),
        "feature_order":       list(prep["feature_names"]),
        "n_features":          prep["n_features"],
        "input_size":          prep["n_features"],
        "sequence_length":     SEQUENCE_LENGTH,
        "aggregation_window":  AGGREGATION_WINDOW_SECONDS,
        "hidden_size":         HIDDEN_SIZE,
        "batch_size":          BATCH_SIZE,
        "learning_rate":       LEARNING_RATE,
        "normalization":       "StandardScaler (fit on training data only)",
        "epochs_trained":      len(history["train_loss"]),
        "seed":                SEED,
        "model_architecture":  "GRU(hidden_size=32) -> Linear(32, n_features)",
        "loss_function":       "MSELoss",
        "optimizer":           "Adam",
        "early_stop_patience": EARLY_STOP_PATIENCE,
        "split_boundary_ts":   float(prep["split_ts"]),
        "forbidden_features":  ["label", "detailedlabel", "ts",
                                 "anomaly_score", "predicted_anomaly"],
    }

    metrics = {
        "training_sequences":    int(len(X_train_raw)),
        "validation_sequences":  int(len(X_val_raw)),
        "final_train_loss":      history["train_loss"][-1],
        "best_val_loss":         min(history["val_loss"]),
        "final_val_loss":        history["val_loss"][-1],
        "normal_validation_trajectory_deviation": {
            "n_sequences":      int(len(val_devs)),
            "mean_deviation":   round(float(val_devs.mean()), 4),
            "median_deviation": round(float(np.median(val_devs)), 4),
            "std_deviation":    round(float(val_devs.std()), 4),
            "p95_deviation":    round(float(p95_normal), 4),
            "p99_deviation":    round(float(np.percentile(val_devs, 99)), 4),
            "max_deviation":    round(float(val_devs.max()), 4),
            "note":             (
                "These statistics describe deviation from expected NORMAL trajectory. "
                "They are dataset-specific. The 95th percentile can serve as a "
                "trajectory anomaly threshold for this dataset only."
            ),
        },
        "malicious_trajectory_deviation": mal_stats if mal_stats else "insufficient_sequences",
        "leakage_checks_passed": True,
        "scaler_fitted_on_train_only": True,
        "chronological_split":   True,
    }

    # -----------------------------------------------------------------------
    # Step 8: Save everything
    # -----------------------------------------------------------------------
    save_artefacts(model, scaler, config, history, metrics, scores_df)

    # -----------------------------------------------------------------------
    # Step 9: Print final report
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  TRINETRA -- GRU Training Complete: Final Report")
    print("=" * 70)

    print(f"\n  GRU input features ({prep['n_features']}):")
    for i, f in enumerate(prep["feature_names"], 1):
        print(f"    {i}. {f}")

    print(f"\n  Aggregation window  : {AGGREGATION_WINDOW_SECONDS} s")
    print(f"  Sequence length     : {SEQUENCE_LENGTH}")
    print(f"  X_train shape       : {X_train_raw.shape}")
    print(f"  y_train shape       : {y_train_raw.shape}")
    print(f"  X_val shape         : {X_val_raw.shape}")
    print(f"  y_val shape         : {y_val_raw.shape}")

    print(f"\n  Model architecture  : GRU(input={prep['n_features']}, hidden=32) -> Linear(32, {prep['n_features']})")
    print(f"  Training samples    : {len(X_train_raw)}")
    print(f"  Validation samples  : {len(X_val_raw)}")
    print(f"  Epochs trained      : {len(history['train_loss'])}")
    print(f"  Final train loss    : {history['train_loss'][-1]:.6f}")
    print(f"  Best val loss       : {min(history['val_loss']):.6f}")
    print(f"  Final val loss      : {history['val_loss'][-1]:.6f}")

    print(f"\n  Normal val trajectory deviation:")
    print(f"    mean   : {val_devs.mean():.4f}")
    print(f"    median : {np.median(val_devs):.4f}")
    print(f"    std    : {val_devs.std():.4f}")
    print(f"    p95    : {p95_normal:.4f}  (dataset-specific threshold)")

    if mal_stats:
        print(f"\n  Malicious trajectory deviation:")
        print(f"    n_seqs : {mal_stats['n_sequences']}")
        print(f"    mean   : {mal_stats['mean_deviation']:.4f}")
        print(f"    median : {mal_stats['median_deviation']:.4f}")
        print(f"    p95    : {mal_stats['p95_deviation']:.4f}")

    print(f"\n  Saved artefacts:")
    print(f"    {GRU_MODEL_PATH}")
    print(f"    {GRU_SCALER_PATH}")
    print(f"    {GRU_CONFIG_PATH}")
    print(f"    {HISTORY_PATH}")
    print(f"    {METRICS_PATH}")
    print(f"    {SCORES_CSV_PATH}")

    print(f"\n  Leakage checks:")
    print(f"    label in features     : {'label' in prep['feature_names']}")
    print(f"    detailedlabel in feat : {'detailedlabel' in prep['feature_names']}")
    print(f"    ts in features        : {'ts' in prep['feature_names']}")
    print(f"    scaler on train only  : True")
    print(f"    chronological split   : True (boundary ts={prep['split_ts']:.3f})")
    print()
