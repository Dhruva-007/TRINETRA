"""
TRINETRA -- train_isolation_forest.py
=====================================
Steps 8-11 & 13:
  - Train/validation/evaluation split using actual labels
  - Fit preprocessing pipeline on NORMAL training data only
  - Train Isolation Forest
  - Evaluate against ground-truth labels
  - Save model, preprocessor, feature schema, and metrics

Training uses only NORMAL (Background + Benign) traffic.
Labels are used ONLY for splitting and post-hoc evaluation.
There is ZERO label leakage into the feature matrix X.

All imports are at the top of the file.
"""

import os
import sys
import json
import logging

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    average_precision_score,
    classification_report,
)
from sklearn.model_selection import train_test_split

from feature_engineering import (
    prepare_feature_matrix,
    build_preprocessor,
    get_all_ml_features,
    get_feature_names_after_transform,
    save_feature_schema,
    validate_required_raw_features,
    COLUMNS_TO_DROP,
    FEATURE_SCHEMA_PATH,
    ML_NUMERICAL_FEATURES,
    ML_CATEGORICAL_FEATURES,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_CSV_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "labeled-conn.csv")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "isolation_forest.joblib")
PREPROCESSOR_PATH = os.path.join(PROJECT_ROOT, "models", "preprocessor.joblib")
FEATURE_NAMES_PATH = os.path.join(PROJECT_ROOT, "models", "feature_names.joblib")
METRICS_PATH = os.path.join(PROJECT_ROOT, "results", "model_metrics.json")
SCORED_CSV_PATH = os.path.join(PROJECT_ROOT, "results", "scored_connections.csv")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Step 8 -- Data loading and train/eval split
# ---------------------------------------------------------------------------

def load_and_split(csv_path: str) -> tuple:
    """
    Load the processed CSV and split into:
      - df_train_normal : Normal (Background + Benign) rows for IF training
      - df_eval         : ALL rows for final evaluation (using ground-truth labels)

    The feature schema is the FIXED ML_NUMERICAL_FEATURES + ML_CATEGORICAL_FEATURES
    defined in feature_engineering.py. It is NOT derived from the live DataFrame.

    Labels are used ONLY for splitting and evaluation -- NEVER as X features.
    Returns: (df_train_normal, df_eval, df_all_eng, num_features, cat_features)
    """
    logger.info("Loading CSV: %s", csv_path)
    df = pd.read_csv(csv_path)
    logger.info("Loaded shape: %s", df.shape)

    label_counts = df["label"].value_counts(dropna=False)
    logger.info("Label distribution:\n%s", label_counts.to_string())

    # Engineer features BEFORE splitting to ensure consistent columns
    df_eng = prepare_feature_matrix(df)

    # Use the FIXED explicit schema (items 4 & 5)
    num_features, cat_features = get_all_ml_features()
    logger.info("Fixed ML schema -- Numerical features (%d): %s", len(num_features), num_features)
    logger.info("Fixed ML schema -- Categorical features (%d): %s", len(cat_features), cat_features)

    # Validate all required features exist before proceeding (items 7 & 8)
    validate_required_raw_features(df_eng, num_features, cat_features)
    logger.info("All required ML features validated OK in full engineered DataFrame.")

    # Split by actual label values
    normal_labels = ["Background", "Benign"]
    df_normal = df_eng[df_eng["label"].isin(normal_labels)].copy()
    df_malicious = df_eng[df_eng["label"] == "Malicious"].copy()

    logger.info(
        "Normal rows (train pool): %d  |  Malicious rows (eval): %d",
        len(df_normal), len(df_malicious)
    )

    # Split normal data: 80% training, 20% validation
    df_train_normal, df_val_normal = train_test_split(
        df_normal, test_size=0.20, random_state=RANDOM_STATE
    )
    logger.info(
        "Train normal: %d  |  Val normal: %d  |  Malicious (eval): %d",
        len(df_train_normal), len(df_val_normal), len(df_malicious)
    )

    # Evaluation set = validation normal + all malicious
    df_eval = pd.concat([df_val_normal, df_malicious], ignore_index=True)
    logger.info("Evaluation set size: %d", len(df_eval))

    return df_train_normal, df_eval, df_eng, num_features, cat_features


# ---------------------------------------------------------------------------
# Step 4 & 5 -- Fit preprocessor on training data
# ---------------------------------------------------------------------------

def fit_preprocessor(df_train: pd.DataFrame,
                      num_features: list,
                      cat_features: list):
    """
    Build and fit the ColumnTransformer on df_train.
    Imputation statistics are derived ONLY from training data.

    After fitting:
      - Verifies preprocessor.get_feature_names_out() matches the saved
        final_feature_names (requirement item 10).
      - Verifies feature dimension matches model.n_features_in_ after
        training (checked in main).

    Returns: (fitted_preprocessor, feature_names_list)
    """
    preprocessor = build_preprocessor(num_features, cat_features)

    logger.info("Fitting preprocessor on %d training rows ...", len(df_train))
    # Pass ONLY the ML features -- never label columns (items 12)
    X_train_raw = df_train[num_features + cat_features]
    preprocessor.fit(X_train_raw)

    # Compute final feature names from the fitted OHE
    feature_names = get_feature_names_after_transform(
        preprocessor, num_features, cat_features
    )
    logger.info("Total features after encoding: %d", len(feature_names))

    # Requirement item 10: verify preprocessor.get_feature_names_out() matches
    raw_pp_names = preprocessor.get_feature_names_out().tolist()
    # Strip "num__" / "cat__" prefix to get plain names
    stripped = [n.split("__", 1)[1] if "__" in n else n for n in raw_pp_names]
    if stripped != feature_names:
        mismatches = [
            (i, feature_names[i], stripped[i])
            for i in range(min(len(feature_names), len(stripped)))
            if feature_names[i] != stripped[i]
        ]
        raise ValueError(
            f"CRITICAL: preprocessor.get_feature_names_out() does NOT match "
            f"the computed feature_names list.\n"
            f"Mismatches ({len(mismatches)}): {mismatches}\n"
            f"  computed : {feature_names}\n"
            f"  from_pp  : {stripped}"
        )
    logger.info(
        "Requirement 10 PASSED: preprocessor.get_feature_names_out() "
        "exactly matches computed feature_names (%d features).",
        len(feature_names),
    )

    return preprocessor, feature_names


def transform_to_matrix(df: pd.DataFrame,
                         preprocessor,
                         num_features: list,
                         cat_features: list) -> np.ndarray:
    """Apply the FITTED preprocessor to a DataFrame and return a numpy array."""
    X_raw = df[num_features + cat_features]
    return preprocessor.transform(X_raw)


# ---------------------------------------------------------------------------
# Step 9 -- Train Isolation Forest
# ---------------------------------------------------------------------------

def train_isolation_forest(X_train: np.ndarray) -> IsolationForest:
    """
    Train an Isolation Forest on normal training data.

    Hyperparameters are fixed as per requirement item 16:
      n_estimators=500, contamination="auto", random_state=42, n_jobs=-1
    """
    logger.info("Training IsolationForest on %d samples x %d features ...",
                X_train.shape[0], X_train.shape[1])

    model = IsolationForest(
        n_estimators=500,
        contamination="auto",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train)
    logger.info("IsolationForest training complete.")
    return model


# ---------------------------------------------------------------------------
# Step 10 -- Evaluate model
# ---------------------------------------------------------------------------

def evaluate_model(model: IsolationForest,
                   X_eval: np.ndarray,
                   df_eval: pd.DataFrame) -> dict:
    """
    Evaluate the trained Isolation Forest against ground-truth labels.

    decision_function() returns the ANOMALY SCORE:
      - More negative  -> more anomalous
      - More positive  -> more normal

    For ROC-AUC (where 1 = anomalous/malicious):
      We negate the score so that higher values = more anomalous.

    predict() returns:
      +1 = inlier  (normal)
      -1 = outlier (anomalous)
    """
    logger.info("Evaluating model on %d eval samples ...", len(df_eval))

    # Ground-truth binary labels (1 = malicious)
    y_true = (df_eval["label"] == "Malicious").astype(int).values

    anomaly_score_raw = model.decision_function(X_eval)
    anomaly_score_negated = -anomaly_score_raw

    y_pred_if = model.predict(X_eval)
    y_pred = (y_pred_if == -1).astype(int)

    roc_auc = roc_auc_score(y_true, anomaly_score_negated)
    pr_auc = average_precision_score(y_true, anomaly_score_negated)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred).tolist()

    metrics = {
        "eval_set_size": int(len(df_eval)),
        "n_malicious_in_eval": int(y_true.sum()),
        "n_normal_in_eval": int((1 - y_true).sum()),
        "roc_auc": round(float(roc_auc), 4),
        "pr_auc": round(float(pr_auc), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
        "confusion_matrix": cm,
        "note": (
            "These metrics measure model performance on THIS dataset only. "
            "Isolation Forest was trained ONLY on normal traffic. "
            "Labels were used ONLY for evaluation -- NOT as training features."
        ),
    }

    logger.info("ROC-AUC          : %.4f", roc_auc)
    logger.info("PR-AUC           : %.4f", pr_auc)
    logger.info("Precision        : %.4f", precision)
    logger.info("Recall           : %.4f", recall)
    logger.info("F1-score         : %.4f", f1)
    logger.info("Confusion matrix : %s", cm)
    logger.info("\n%s", classification_report(y_true, y_pred,
                                              target_names=["Normal", "Malicious"]))

    return metrics, anomaly_score_raw, y_pred


# ---------------------------------------------------------------------------
# Step 11 -- Save all artefacts
# ---------------------------------------------------------------------------

def save_artefacts(model, preprocessor, num_features, cat_features,
                   feature_names, metrics, df_eval, anomaly_score_raw, y_pred):
    """
    Save model, preprocessor, feature schema (three parts), metrics, and scored CSV.

    Requirement item 6: saves
      - raw ordered numerical feature names
      - raw ordered categorical feature names
      - final transformed feature names from fitted OHE
    """
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)

    joblib.dump(model, MODEL_PATH)
    logger.info("Model saved        : %s", MODEL_PATH)

    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    logger.info("Preprocessor saved : %s", PREPROCESSOR_PATH)

    # Legacy flat list for any old code that loads feature_names.joblib
    joblib.dump(feature_names, FEATURE_NAMES_PATH)
    logger.info("Feature names saved: %s", FEATURE_NAMES_PATH)

    # Full schema with all three parts (requirement item 6)
    save_feature_schema(num_features, cat_features, feature_names, FEATURE_SCHEMA_PATH)

    with open(METRICS_PATH, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    logger.info("Metrics saved      : %s", METRICS_PATH)

    # Scored connections CSV (labels kept for evaluation context, item 13)
    df_scored = df_eval[
        ["ts", "uid", "id.orig_h", "id.orig_p", "id.resp_h", "id.resp_p",
         "proto", "service", "label", "detailedlabel"]
    ].copy()
    df_scored["anomaly_score_raw"] = anomaly_score_raw
    df_scored["anomaly_score_negated"] = -anomaly_score_raw
    df_scored["predicted_anomaly"] = y_pred  # 1 = anomaly, 0 = normal
    df_scored.to_csv(SCORED_CSV_PATH, index=False)
    logger.info("Scored CSV saved   : %s", SCORED_CSV_PATH)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("TRINETRA -- Isolation Forest Training Pipeline")
    logger.info("=" * 60)

    # Step 8 -- Load and split
    df_train_normal, df_eval, df_all_eng, num_features, cat_features = load_and_split(
        PROCESSED_CSV_PATH
    )

    # Step 4 & 5 -- Fit preprocessor on training data only
    preprocessor, feature_names = fit_preprocessor(
        df_train_normal, num_features, cat_features
    )

    # Transform training and evaluation sets
    X_train = transform_to_matrix(df_train_normal, preprocessor, num_features, cat_features)
    X_eval = transform_to_matrix(df_eval, preprocessor, num_features, cat_features)

    logger.info("X_train shape: %s", X_train.shape)
    logger.info("X_eval shape : %s", X_eval.shape)

    # Step 9 -- Train Isolation Forest
    model = train_isolation_forest(X_train)

    # Requirement item 11: verify transformed feature dimension matches model
    if X_train.shape[1] != model.n_features_in_:
        raise ValueError(
            f"CRITICAL: X_train has {X_train.shape[1]} features but "
            f"model.n_features_in_ = {model.n_features_in_}. "
            "Re-run training to regenerate consistent artefacts."
        )
    if len(feature_names) != model.n_features_in_:
        raise ValueError(
            f"CRITICAL: feature_names has {len(feature_names)} entries but "
            f"model.n_features_in_ = {model.n_features_in_}."
        )
    logger.info(
        "Requirement 11 PASSED: feature dimension %d matches model.n_features_in_ = %d.",
        X_train.shape[1], model.n_features_in_,
    )

    # Step 10 -- Evaluate
    metrics, anomaly_score_raw, y_pred = evaluate_model(model, X_eval, df_eval)

    # Step 11 -- Save everything
    save_artefacts(
        model, preprocessor, num_features, cat_features,
        feature_names, metrics, df_eval, anomaly_score_raw, y_pred
    )

    # -----------------------------------------------------------------------
    # Print required schema report (items B, C, D)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  TRINETRA -- ML Feature Schema Report")
    print("=" * 70)

    print(f"\nB. Raw numerical ML features ({len(num_features)}):")
    for i, f in enumerate(num_features, 1):
        print(f"   {i:2d}. {f}")

    print(f"\n   Raw categorical ML features ({len(cat_features)}):")
    for i, f in enumerate(cat_features, 1):
        print(f"   {i:2d}. {f}")

    print(f"\nC. Final transformed feature names ({len(feature_names)}):")
    for i, f in enumerate(feature_names, 1):
        print(f"   {i:2d}. {f}")

    print(f"\nD. Total feature count: {len(feature_names)}")
    print(f"   model.n_features_in_ = {model.n_features_in_}")
    print(f"   Match: {len(feature_names) == model.n_features_in_}")

    logger.info("=" * 60)
    logger.info("TRINETRA pipeline complete.")
    logger.info("=" * 60)
