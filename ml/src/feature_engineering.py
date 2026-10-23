"""
TRINETRA -- feature_engineering.py
===================================
Steps 3-7: Feature identification, missing-value handling,
           categorical encoding, feature engineering, and
           column dropping for Isolation Forest.

All imports are at the top of the file.
"""

import os
import sys
import logging

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import joblib

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_CSV_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "labeled-conn.csv")
PREPROCESSOR_PATH = os.path.join(PROJECT_ROOT, "models", "preprocessor.joblib")
FEATURE_NAMES_PATH = os.path.join(PROJECT_ROOT, "models", "feature_names.joblib")
FEATURE_SCHEMA_PATH = os.path.join(PROJECT_ROOT, "models", "feature_schema.joblib")

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
# Step 3 -- Feature identification
# ---------------------------------------------------------------------------

# Columns to drop (never enter the ML feature matrix)
COLUMNS_TO_DROP = [
    "uid",            # Row identifier -- carries no behavioural signal
    "id.orig_h",      # Source IP -- identifier, not a behaviour feature
    "id.resp_h",      # Destination IP -- identifier, same reason
    "ts",             # Timestamp -- temporal identifier; data-ordering leakage
    "tunnel_parents", # 100% empty in this dataset -> zero information content
    "local_orig",     # Boolean flag; nearly constant and IP-topology-dependent
    "local_resp",     # Boolean flag; nearly constant and IP-topology-dependent
    "label",          # TARGET -- must never be an input feature
    "detailedlabel",  # TARGET detail -- same reason
]

# ---------------------------------------------------------------------------
# Explicit ML Feature Schema  (requirement items 4 & 5)
# ---------------------------------------------------------------------------
# One canonical source of truth for what enters the model.
# Never derived from the live DataFrame columns.
# Training and inference BOTH reference these lists -- no silent changes.

# 10 raw Zeek numerical + 6 engineered = 16 numerical total
ML_NUMERICAL_FEATURES = [
    "id.orig_p",           # Source port
    "id.resp_p",           # Destination port
    "duration",            # Connection duration (40.8% missing -> impute)
    "orig_bytes",          # Bytes sent by originator (40.8% missing -> impute)
    "resp_bytes",          # Bytes sent by responder (40.8% missing -> impute)
    "missed_bytes",        # Bytes missed by Zeek
    "orig_pkts",           # Packets sent by originator
    "orig_ip_bytes",       # IP-level bytes from originator
    "resp_pkts",           # Packets sent by responder
    "resp_ip_bytes",       # IP-level bytes from responder
    # 6 engineered behavioural features
    "packet_count",        # orig_pkts + resp_pkts
    "total_bytes",         # orig_bytes + resp_bytes
    "bytes_per_packet",    # total_bytes / packet_count
    "packet_rate",         # packet_count / duration
    "byte_rate",           # total_bytes / duration
    "sent_received_ratio", # orig_bytes / resp_bytes (NaN when resp_bytes is NaN)
]  # 16 numerical features total

# 4 categorical source features
ML_CATEGORICAL_FEATURES = [
    "proto",       # Transport protocol: tcp, udp, icmp
    "service",     # Application service: dns, http, ssl (57.4% missing -> impute)
    "conn_state",  # Connection state machine flag
    "history",     # TCP flag history string (32.6% missing -> impute)
]  # 4 categorical features total

# Backwards-compatible aliases
NUMERICAL_FEATURES = ML_NUMERICAL_FEATURES
CATEGORICAL_FEATURES = ML_CATEGORICAL_FEATURES

# Columns kept for context / split logic but NOT used in ML
CONTEXT_COLUMNS = [
    "id.orig_h",
    "id.resp_h",
    "ts",
    "label",
    "detailedlabel",
]

# ---------------------------------------------------------------------------
# Step 6 -- Feature engineering helpers
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive behavioural network features from raw Zeek connection fields.

    NaN propagates naturally -- no fillna(0) before division (requirement 3).
    The numerical imputer handles resulting NaN values.
    All divisions are protected against zero-division and infinity.

    Returns a new DataFrame with additional engineered columns.
    Does NOT modify the input DataFrame in place.
    """
    df = df.copy()
    required_cols = set(df.columns)

    # 1. packet_count = orig_pkts + resp_pkts
    if {"orig_pkts", "resp_pkts"}.issubset(required_cols):
        df["packet_count"] = df["orig_pkts"] + df["resp_pkts"]
        logger.info("  [+] packet_count created")
    else:
        logger.warning("  [-] packet_count skipped -- missing orig_pkts or resp_pkts")

    # 2. total_bytes = orig_bytes + resp_bytes
    if {"orig_bytes", "resp_bytes"}.issubset(required_cols):
        df["total_bytes"] = df["orig_bytes"] + df["resp_bytes"]
        logger.info("  [+] total_bytes created")
    else:
        logger.warning("  [-] total_bytes skipped -- missing orig_bytes or resp_bytes")

    # 3. bytes_per_packet = total_bytes / packet_count  (protect /0)
    if {"total_bytes", "packet_count"}.issubset(df.columns):
        df["bytes_per_packet"] = df["total_bytes"] / df["packet_count"].replace(0, np.nan)
        df["bytes_per_packet"] = df["bytes_per_packet"].replace([np.inf, -np.inf], np.nan)
        logger.info("  [+] bytes_per_packet created")

    # 4. packet_rate = packet_count / duration  (protect /0 and /NaN)
    if {"packet_count", "duration"}.issubset(df.columns):
        df["packet_rate"] = df["packet_count"] / df["duration"].replace(0, np.nan)
        df["packet_rate"] = df["packet_rate"].replace([np.inf, -np.inf], np.nan)
        logger.info("  [+] packet_rate created")
    else:
        logger.warning("  [-] packet_rate skipped -- missing packet_count or duration")

    # 5. byte_rate = total_bytes / duration  (protect /0 and /NaN)
    if {"total_bytes", "duration"}.issubset(df.columns):
        df["byte_rate"] = df["total_bytes"] / df["duration"].replace(0, np.nan)
        df["byte_rate"] = df["byte_rate"].replace([np.inf, -np.inf], np.nan)
        logger.info("  [+] byte_rate created")
    else:
        logger.warning("  [-] byte_rate skipped -- missing total_bytes or duration")

    # 6. sent_received_ratio = orig_bytes / resp_bytes
    #    NaN propagates when resp_bytes is NaN (requirement item 3).
    #    Replace resp_bytes == 0 with NaN to avoid inf.
    #    The numerical imputer handles the resulting NaN.
    if {"orig_bytes", "resp_bytes"}.issubset(required_cols):
        df["sent_received_ratio"] = df["orig_bytes"] / df["resp_bytes"].replace(0, np.nan)
        df["sent_received_ratio"] = df["sent_received_ratio"].replace([np.inf, -np.inf], np.nan)
        logger.info("  [+] sent_received_ratio created (NaN where resp_bytes is NaN or 0)")
    else:
        logger.warning("  [-] sent_received_ratio skipped -- missing orig_bytes or resp_bytes")

    return df


# ---------------------------------------------------------------------------
# Step 4 & 5 -- Build preprocessing pipeline
# ---------------------------------------------------------------------------

def build_preprocessor(numerical_features: list, categorical_features: list) -> ColumnTransformer:
    """
    Build a sklearn ColumnTransformer that:
      - Imputes numerical features with mean (fit on training data only)
      - Imputes categorical features with most-frequent value
      - One-hot encodes categorical features (handle_unknown='ignore')

    No scaling -- Isolation Forest does not require it (requirement 15).
    """
    numerical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, numerical_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )

    return preprocessor


def get_feature_names_after_transform(preprocessor: ColumnTransformer,
                                      numerical_features: list,
                                      categorical_features: list) -> list:
    """
    Extract the final ordered feature names after the ColumnTransformer
    has been fitted. Numerical features keep their names; one-hot encoded
    features get names like proto_tcp, service_dns, etc.
    """
    ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    ohe_feature_names = ohe.get_feature_names_out(categorical_features).tolist()
    return numerical_features + ohe_feature_names


def prepare_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given the raw (post-CSV-load) DataFrame, engineer behavioural features
    and return the augmented DataFrame. Label/identifier columns remain
    present -- they are excluded by the caller.
    """
    logger.info("Engineering behavioural features ...")
    df_eng = engineer_features(df)
    return df_eng


def get_all_ml_features(df_engineered=None) -> tuple:
    """
    Return the fixed (ML_NUMERICAL_FEATURES, ML_CATEGORICAL_FEATURES) schema.

    The explicit schema is NEVER derived from whatever columns happen to
    exist in the input DataFrame (requirement items 4 & 5).
    The df_engineered parameter is kept for backwards-compatibility but
    is intentionally unused.
    """
    return list(ML_NUMERICAL_FEATURES), list(ML_CATEGORICAL_FEATURES)


# ---------------------------------------------------------------------------
# Schema persistence helpers  (requirement items 6 & 7)
# ---------------------------------------------------------------------------

def save_feature_schema(numerical_features: list,
                         categorical_features: list,
                         final_feature_names: list,
                         schema_path: str) -> None:
    """
    Persist the three-part ML feature schema to disk:
      - numerical_features   : ordered raw numerical feature names (16)
      - categorical_features : ordered raw categorical feature names (4)
      - final_feature_names  : ordered post-transform feature names from OHE

    This is the single authoritative record of what the model was trained on.
    """
    schema = {
        "numerical_features": list(numerical_features),
        "categorical_features": list(categorical_features),
        "final_feature_names": list(final_feature_names),
    }
    os.makedirs(os.path.dirname(schema_path), exist_ok=True)
    joblib.dump(schema, schema_path)
    logger.info("Feature schema saved: %s", schema_path)
    logger.info("  numerical  (%d): %s", len(numerical_features), numerical_features)
    logger.info("  categorical (%d): %s", len(categorical_features), categorical_features)
    logger.info("  final (post-transform, %d): %s", len(final_feature_names), final_feature_names)


def load_feature_schema(schema_path: str) -> dict:
    """
    Load the saved ML feature schema from disk.

    Returns a dict with keys:
      "numerical_features"   -> list[str]
      "categorical_features" -> list[str]
      "final_feature_names"  -> list[str]

    Raises FileNotFoundError if the schema has not been saved yet.
    """
    if not os.path.exists(schema_path):
        raise FileNotFoundError(
            f"Feature schema not found: {schema_path}\n"
            "Run train_isolation_forest.py first to generate it."
        )
    schema = joblib.load(schema_path)
    logger.info(
        "Feature schema loaded: %d numerical, %d categorical, %d final features",
        len(schema["numerical_features"]),
        len(schema["categorical_features"]),
        len(schema["final_feature_names"]),
    )
    return schema


def validate_required_raw_features(df_eng: pd.DataFrame,
                                    numerical_features: list,
                                    categorical_features: list) -> None:
    """
    Verify every required ML feature is present in the engineered DataFrame
    BEFORE passing it to the preprocessor.

    - Extra columns are silently ignored (requirement item 9).
    - Missing columns raise ValueError listing all absent features (item 8).
    - Label columns are excluded from the required list (item 12).
    """
    _LABEL_COLS = {"label", "detailedlabel"}
    required = [
        f for f in (numerical_features + categorical_features)
        if f not in _LABEL_COLS
    ]
    df_cols = set(df_eng.columns)
    missing = [f for f in required if f not in df_cols]
    if missing:
        raise ValueError(
            f"Input DataFrame is missing {len(missing)} required ML feature(s). "
            f"Missing: {missing}\n"
            f"Available columns: {sorted(df_cols)}\n"
            "Ensure feature engineering ran successfully and that the input "
            "data contains the necessary source columns."
        )
    extra = df_cols - set(required)
    if extra:
        logger.info(
            "%d extra column(s) present and will be ignored: %s",
            len(extra), sorted(extra),
        )


# ---------------------------------------------------------------------------
# Entry point (self-test)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info("Loading processed CSV for feature engineering self-test ...")
    df = pd.read_csv(PROCESSED_CSV_PATH)

    logger.info("Raw shape: %s", df.shape)
    df_eng = prepare_feature_matrix(df)
    logger.info("Engineered shape: %s", df_eng.shape)

    num_feats, cat_feats = get_all_ml_features()
    logger.info("Numerical features (%d): %s", len(num_feats), num_feats)
    logger.info("Categorical features (%d): %s", len(cat_feats), cat_feats)
    logger.info("Columns to drop: %s", COLUMNS_TO_DROP)

    validate_required_raw_features(df_eng, num_feats, cat_feats)
    logger.info("All required ML features are present in the engineered DataFrame.")
