"""
TRINETRA -- inference.py
=========================
Apply the trained Isolation Forest + preprocessing pipeline to new
Zeek connection data (either a preprocessed CSV or a raw Zeek TXT file).

Design contract
---------------
* feature_schema.joblib is the source of truth for the ML feature schema:
    - numerical_features   : 16 ordered raw numerical feature names
    - categorical_features : 4  ordered raw categorical feature names
    - final_feature_names  : ordered post-transform feature names
* The schema is loaded at startup and used to validate every input.
* Inference NEVER re-derives the feature list from the live DataFrame.
* label and detailedlabel are never included in X under any circumstance.
* anomaly_score_raw is the raw decision_function output -- NOT a probability.
* Extra raw Zeek columns are allowed and simply ignored (requirement 9).
* Missing required features raise a clear ValueError (requirement 8).

All imports are at the top of the file.
"""

import argparse
import os
import sys
import logging

import numpy as np
import pandas as pd
import joblib

from data_preprocessing import load_zeek_to_dataframe, infer_and_cast_types
from feature_engineering import (
    prepare_feature_matrix,
    load_feature_schema,
    validate_required_raw_features,
    FEATURE_SCHEMA_PATH,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "isolation_forest.joblib")
PREPROCESSOR_PATH = os.path.join(PROJECT_ROOT, "models", "preprocessor.joblib")
FEATURE_NAMES_PATH = os.path.join(PROJECT_ROOT, "models", "feature_names.joblib")

# Columns that must NEVER enter the feature matrix X regardless of presence
_LABEL_COLUMNS = {"label", "detailedlabel"}

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
# Artefact loading
# ---------------------------------------------------------------------------

def load_model_artefacts() -> tuple:
    """
    Load the trained Isolation Forest, the fitted ColumnTransformer,
    and the saved feature schema from disk.

    Returns
    -------
    model        : trained IsolationForest
    preprocessor : fitted ColumnTransformer
    schema       : dict with keys numerical_features, categorical_features,
                   final_feature_names -- the authoritative training schema
    """
    for path in (MODEL_PATH, PREPROCESSOR_PATH, FEATURE_SCHEMA_PATH):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required artefact not found: {path}")

    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    schema = load_feature_schema(FEATURE_SCHEMA_PATH)

    logger.info("Model loaded         : %s", MODEL_PATH)
    logger.info("Preprocessor loaded  : %s", PREPROCESSOR_PATH)
    logger.info(
        "Schema loaded        : %d numerical, %d categorical, %d final features",
        len(schema["numerical_features"]),
        len(schema["categorical_features"]),
        len(schema["final_feature_names"]),
    )

    return model, preprocessor, schema


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _normalize_feature_names(raw_names: list) -> list:
    """
    Strip the transformer-prefix that sklearn ColumnTransformer prepends in
    get_feature_names_out() results (e.g. "num__id.orig_p" -> "id.orig_p",
    "cat__proto_icmp" -> "proto_icmp").
    """
    normalized = []
    for name in raw_names:
        if "__" in name:
            normalized.append(name.split("__", 1)[1])
        else:
            normalized.append(name)
    return normalized


def validate_artefact_consistency(preprocessor, schema: dict) -> None:
    """
    Cross-validate the saved feature_schema final_feature_names against
    the names the fitted preprocessor produces via get_feature_names_out()
    (requirement item 10).

    Also checks that the final feature count matches model.n_features_in_
    (requirement item 11 -- called after model is loaded by the caller).

    Raises ValueError if count or any individual name disagrees.
    """
    final_feature_names = schema["final_feature_names"]

    try:
        raw_pp_names = preprocessor.get_feature_names_out().tolist()
    except AttributeError:
        logger.warning(
            "preprocessor.get_feature_names_out() not available on this "
            "sklearn version -- skipping artefact cross-validation."
        )
        return

    preprocessor_names = _normalize_feature_names(raw_pp_names)

    if len(preprocessor_names) != len(final_feature_names):
        raise ValueError(
            f"Artefact inconsistency: feature_schema has "
            f"{len(final_feature_names)} final_feature_names but the fitted "
            f"preprocessor produces {len(preprocessor_names)} output columns "
            f"after stripping transformer prefixes."
        )

    mismatches = [
        (i, final_feature_names[i], preprocessor_names[i])
        for i in range(len(final_feature_names))
        if final_feature_names[i] != preprocessor_names[i]
    ]
    if mismatches:
        detail = "\n".join(
            f"  index {i}: saved='{saved}'  preprocessor='{from_pp}'"
            for i, saved, from_pp in mismatches
        )
        raise ValueError(
            f"Artefact inconsistency: {len(mismatches)} feature name(s) "
            f"disagree between feature_schema.final_feature_names and the "
            f"fitted preprocessor (after stripping transformer prefixes):\n{detail}"
        )

    logger.info(
        "Requirement 10 PASSED (inference): all %d feature names agree "
        "between feature_schema and preprocessor.get_feature_names_out().",
        len(final_feature_names),
    )


def validate_transform_output(X: np.ndarray, model, schema: dict) -> None:
    """
    After preprocessing, verify that:
      1. The transformed matrix has exactly len(final_feature_names) columns.
      2. The column count matches model.n_features_in_.
    (Requirement item 11.)

    Raises ValueError on any mismatch so the pipeline fails fast.
    """
    final_feature_names = schema["final_feature_names"]
    n_expected_schema = len(final_feature_names)
    n_model = model.n_features_in_

    if X.shape[1] != n_expected_schema:
        raise ValueError(
            f"Transformed feature matrix has {X.shape[1]} column(s) but "
            f"feature_schema expects {n_expected_schema}. "
            "Re-run training to regenerate consistent artefacts."
        )
    if X.shape[1] != n_model:
        raise ValueError(
            f"Transformed feature matrix has {X.shape[1]} column(s) but "
            f"model.n_features_in_ = {n_model}. "
            "Re-run training to regenerate consistent artefacts."
        )
    logger.info(
        "Requirement 11 PASSED (inference): transform dimension %d matches "
        "feature_schema (%d) and model.n_features_in_ (%d).",
        X.shape[1], n_expected_schema, n_model,
    )


# ---------------------------------------------------------------------------
# Core inference
# ---------------------------------------------------------------------------

def run_inference_on_dataframe(df: pd.DataFrame,
                                model,
                                preprocessor,
                                schema: dict) -> pd.DataFrame:
    """
    Score a DataFrame of Zeek connection records using the trained
    Isolation Forest.

    Pipeline (mirrors training exactly):
      1. Cast column types.
      2. Engineer behavioural features (identical functions to training).
      3. Load required raw feature lists from the saved schema (NOT from
         get_all_ml_features() applied to the live DataFrame).
      4. Validate all required raw features exist BEFORE transform() (items 7 & 8).
      5. Pass only the required columns in the schema-defined order.
      6. Transform (no re-fitting).
      7. Validate output dimension (items 10 & 11).
      8. Score with decision_function() and predict().

    label / detailedlabel are excluded from X throughout (items 12 & 13).
    Extra raw Zeek columns are ignored (item 9).
    """
    num_features = schema["numerical_features"]
    cat_features = schema["categorical_features"]
    final_feature_names = schema["final_feature_names"]

    # -- Step 1: type casting ------------------------------------------------
    df = infer_and_cast_types(df)

    # -- Step 2: feature engineering -----------------------------------------
    df_eng = prepare_feature_matrix(df)

    # -- Step 3: log the schema used -----------------------------------------
    logger.info(
        "Schema -- required numerical (%d): %s",
        len(num_features), num_features,
    )
    logger.info(
        "Schema -- required categorical (%d): %s",
        len(cat_features), cat_features,
    )

    # -- Step 4: validate required raw features exist (items 7 & 8) ----------
    validate_required_raw_features(df_eng, num_features, cat_features)

    # Hard guard: label columns must never enter the feature matrix (item 12)
    for bad_col in _LABEL_COLUMNS:
        if bad_col in (num_features + cat_features):
            raise RuntimeError(
                f"CRITICAL: label column '{bad_col}' is present in the "
                f"feature schema. The saved schema was built incorrectly. "
                "Re-run training."
            )

    # -- Step 5: build the raw input matrix in schema-defined column order ----
    X_raw = df_eng[num_features + cat_features]

    # -- Step 6: transform (no re-fitting) ------------------------------------
    X = preprocessor.transform(X_raw)

    # -- Step 7: validate output dimension (items 10 & 11) -------------------
    validate_transform_output(X, model, schema)

    # -- Step 8: score --------------------------------------------------------
    anomaly_score_raw = model.decision_function(X)
    anomaly_score_negated = -anomaly_score_raw
    y_pred_if = model.predict(X)
    predicted_anomaly = (y_pred_if == -1).astype(int)

    # -- Attach results to a copy of the original DataFrame ------------------
    result_df = df.copy()
    result_df["anomaly_score_raw"] = anomaly_score_raw
    result_df["anomaly_score_negated"] = anomaly_score_negated
    result_df["predicted_anomaly"] = predicted_anomaly

    n_total = len(result_df)
    n_anomalies = int(predicted_anomaly.sum())
    pct = 100.0 * n_anomalies / max(n_total, 1)
    logger.info(
        "Inference complete: %d rows scored | %d flagged anomalous (%.2f%%)",
        n_total, n_anomalies, pct,
    )

    return result_df


def run_inference_on_csv(csv_path: str) -> pd.DataFrame:
    """
    Run inference on a preprocessed CSV file (same schema as labeled-conn.csv).
    label / detailedlabel, if present, are kept in the output for context
    but are never passed to the preprocessor (items 12 & 13).
    """
    logger.info("Loading CSV for inference: %s", csv_path)
    df = pd.read_csv(csv_path)
    model, preprocessor, schema = load_model_artefacts()
    validate_artefact_consistency(preprocessor, schema)
    return run_inference_on_dataframe(df, model, preprocessor, schema)


def run_inference_on_zeek_txt(txt_path: str) -> pd.DataFrame:
    """
    Run inference directly on a raw Zeek conn.log TXT file.
    Parses the actual #fields header, replaces "-" / "(empty)" sentinels
    with NaN, then applies the full pipeline.
    """
    logger.info("Loading Zeek TXT for inference: %s", txt_path)
    df = load_zeek_to_dataframe(txt_path)
    model, preprocessor, schema = load_model_artefacts()
    validate_artefact_consistency(preprocessor, schema)
    return run_inference_on_dataframe(df, model, preprocessor, schema)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="TRINETRA -- Isolation Forest inference on Zeek connection logs"
    )
    parser.add_argument(
        "input_path",
        help=(
            "Path to input file. Accepts:\n"
            "  - a preprocessed CSV (same schema as labeled-conn.csv)\n"
            "  - a raw Zeek conn.log TXT file (*.txt or *.log)"
        ),
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Optional path to write scored results CSV.",
    )
    return parser


if __name__ == "__main__":
    parser = _build_arg_parser()
    args = parser.parse_args()
    input_path = args.input_path

    if input_path.endswith(".txt") or input_path.endswith(".log"):
        result_df = run_inference_on_zeek_txt(input_path)
    else:
        result_df = run_inference_on_csv(input_path)

    # Report key statistics
    n_total = len(result_df)
    n_anomalies = int(result_df["predicted_anomaly"].sum())
    pct = 100.0 * n_anomalies / max(n_total, 1)

    print("\n" + "=" * 60)
    print("  TRINETRA -- Inference Summary")
    print("=" * 60)
    print(f"  Rows scored           : {n_total}")
    print(f"  Flagged as anomalous  : {n_anomalies}  ({pct:.2f}%)")
    print(f"  Normal (not flagged)  : {n_total - n_anomalies}")

    # Show top anomalies
    top = result_df.sort_values("anomaly_score_negated", ascending=False).head(20)
    display_cols = [
        c for c in [
            "ts", "uid", "id.orig_h", "id.orig_p",
            "id.resp_h", "id.resp_p", "proto", "service",
            "anomaly_score_negated", "predicted_anomaly", "label",
        ]
        if c in result_df.columns
    ]
    print("\n--- TOP 20 MOST ANOMALOUS CONNECTIONS ---")
    print(top[display_cols].to_string(index=False))

    if args.output:
        result_df.to_csv(args.output, index=False)
        logger.info("Scored results written to: %s", args.output)
    else:
        logger.info("No --output path specified; scored results not saved to disk.")
