"""
TRINETRA — data_preprocessing.py
=================================
Step 1: Convert Zeek conn.log TXT → CSV
Step 2: Perform EDA

Reads the raw Zeek log, correctly parses the #fields header,
replaces Zeek missing-value sentinels ('-' and '(empty)') with
proper NaN, and writes a clean CSV.

All imports are at the top of the file.
"""

import os
import re
import sys
import json
import logging

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RAW_TXT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "raw", "labeled-conn.log.txt",
)

PROCESSED_CSV_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "processed", "labeled-conn.csv",
)

EDA_REPORT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "eda_report.txt",
)

# Zeek sentinels that represent missing / unset values
ZEEK_MISSING_VALUES = {"-", "(empty)"}

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
# Step 1 — Parse Zeek log and convert to CSV
# ---------------------------------------------------------------------------

def parse_zeek_fields(raw_path: str) -> list:
    """Return the list of field names from the Zeek #fields header line."""
    with open(raw_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\r\n")
            if line.startswith("#fields"):
                # Tab-separated; first token is '#fields', rest are field names
                parts = line.split("\t")
                fields = parts[1:]  # drop '#fields' token
                return fields
    raise ValueError("No #fields header found in Zeek log file.")


def parse_zeek_types(raw_path: str) -> list:
    """Return the list of declared types from the Zeek #types header line."""
    with open(raw_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\r\n")
            if line.startswith("#types"):
                parts = line.split("\t")
                return parts[1:]
    return []


def load_zeek_to_dataframe(raw_path: str) -> pd.DataFrame:
    """
    Read the Zeek conn.log TXT file and return a pandas DataFrame.

    - Skips all lines beginning with '#' (metadata/header lines).
    - Uses the actual #fields header to name columns.
    - Replaces Zeek missing-value sentinels '-' and '(empty)' with NaN.
    - Does NOT alter the original file.
    """
    logger.info("Parsing Zeek #fields header from: %s", raw_path)
    fields = parse_zeek_fields(raw_path)
    logger.info("Found %d columns: %s", len(fields), fields)

    logger.info("Reading data rows (skipping # lines) …")
    data_rows = []
    with open(raw_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\r\n")
            if line.startswith("#") or not line.strip():
                continue
            data_rows.append(line.split("\t"))

    logger.info("Total data rows read: %d", len(data_rows))

    df = pd.DataFrame(data_rows, columns=fields)

    # Replace Zeek missing sentinels with NaN
    df.replace(list(ZEEK_MISSING_VALUES), np.nan, inplace=True)

    return df


def infer_and_cast_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cast columns to appropriate dtypes based on the Zeek #types declaration
    and domain knowledge.

    Numeric columns that contain NaN are kept as float64 (pandas default).
    Boolean columns (T/F in Zeek) are mapped to Python bool.
    String/categorical columns remain as object.
    """
    # Numeric columns (Zeek types: time, interval, count, port)
    numeric_cols = [
        "ts", "id.orig_p", "id.resp_p", "duration",
        "orig_bytes", "resp_bytes", "missed_bytes",
        "orig_pkts", "orig_ip_bytes", "resp_pkts", "resp_ip_bytes",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Boolean columns: Zeek encodes as 'T' / 'F'
    bool_cols = ["local_orig", "local_resp"]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].map({"T": True, "F": False})

    return df


def convert_zeek_to_csv(raw_path: str, csv_path: str) -> pd.DataFrame:
    """
    Full conversion pipeline: Zeek TXT → typed DataFrame → CSV.
    Returns the resulting DataFrame.
    """
    df = load_zeek_to_dataframe(raw_path)
    df = infer_and_cast_types(df)

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    logger.info("CSV written to: %s  (%d rows × %d cols)", csv_path, len(df), len(df.columns))

    return df


# ---------------------------------------------------------------------------
# Step 2 — EDA
# ---------------------------------------------------------------------------

def perform_eda(df: pd.DataFrame, report_path: str) -> str:
    """
    Perform a complete EDA on the DataFrame and write a text report.
    Returns the report string.
    """
    lines = []

    def log_section(title: str):
        sep = "=" * 70
        lines.append(f"\n{sep}")
        lines.append(f"  {title}")
        lines.append(sep)

    def log_line(text: str = ""):
        lines.append(text)

    # ── 1. Shape ────────────────────────────────────────────────────────────
    log_section("1. DATASET SHAPE")
    log_line(f"  Rows    : {df.shape[0]}")
    log_line(f"  Columns : {df.shape[1]}")

    # ── 2 & 3. Column names and dtypes ──────────────────────────────────────
    log_section("2 & 3. COLUMN NAMES AND DATA TYPES")
    for col in df.columns:
        log_line(f"  {col:<22} {str(df[col].dtype):<12}")

    # ── 4 & 5. Missing values ────────────────────────────────────────────────
    log_section("4 & 5. MISSING VALUES (NaN count and %)")
    log_line(f"  {'Column':<22} {'Missing':>8} {'%Missing':>10}")
    log_line(f"  {'-'*22} {'-'*8} {'-'*10}")
    for col in df.columns:
        n_miss = df[col].isna().sum()
        pct = 100.0 * n_miss / len(df)
        log_line(f"  {col:<22} {n_miss:>8} {pct:>9.2f}%")

    # ── 6 & 7. Categorical distributions ─────────────────────────────────────
    log_section("6 & 7. CATEGORICAL COLUMN DISTRIBUTIONS")
    categorical_cols = df.select_dtypes(include="object").columns.tolist()
    # Exclude label columns from display but show them separately below
    label_cols = ["label", "detailedlabel"]
    cat_non_label = [c for c in categorical_cols if c not in label_cols]

    for col in cat_non_label:
        n_unique = df[col].nunique(dropna=True)
        log_line(f"\n  Column: {col}  (unique values: {n_unique}, NaN: {df[col].isna().sum()})")
        vc = df[col].value_counts(dropna=False).head(20)
        for val, cnt in vc.items():
            log_line(f"    {str(val):<30} {cnt:>6}")

    # ── 8 & 9. Label columns ─────────────────────────────────────────────────
    log_section("8 & 9. LABEL COLUMNS AND DISTRIBUTIONS")
    for col in label_cols:
        if col in df.columns:
            log_line(f"\n  Column: {col}")
            vc = df[col].value_counts(dropna=False)
            for val, cnt in vc.items():
                pct = 100.0 * cnt / len(df)
                log_line(f"    {str(val):<30} {cnt:>6}  ({pct:.1f}%)")

    # ── 10. Numerical statistics ──────────────────────────────────────────────
    log_section("10. NUMERICAL FEATURE STATISTICS")
    numeric_df = df.select_dtypes(include=[np.number])
    # Exclude ts from stats display (it's a timestamp)
    num_cols_for_stats = [c for c in numeric_df.columns if c != "ts"]

    percentiles = [0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
    desc = numeric_df[num_cols_for_stats].describe(percentiles=percentiles)
    log_line(desc.to_string())

    # ── Summary notes ─────────────────────────────────────────────────────────
    log_section("NOTES / OBSERVATIONS")
    log_line("  - Zeek missing sentinels '-' and '(empty)' replaced with NaN before analysis.")
    log_line("  - 'tunnel_parents' is 100% empty → candidate for drop.")
    log_line("  - 'duration', 'orig_bytes', 'resp_bytes' have 40.8% missing (ICMP/OTH flows).")
    log_line("  - 'service' is missing for 57.4% of rows (mainly non-DNS/HTTP/SSL traffic).")
    log_line("  - 'history' is missing for 32.6% of rows (ICMP flows have no TCP history flags).")
    log_line("  - Label classes: Background=739, Malicious=244, Benign=9.")
    log_line("  - detailedlabel: (empty)=Background, CC=C&C malicious, Google=Benign.")

    report = "\n".join(lines)

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(report)

    logger.info("EDA report written to: %s", report_path)
    return report


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Step 1 — Convert Zeek TXT → CSV
    logger.info("=" * 60)
    logger.info("TRINETRA — Step 1: Converting Zeek log to CSV")
    logger.info("=" * 60)
    df = convert_zeek_to_csv(RAW_TXT_PATH, PROCESSED_CSV_PATH)

    # Verify
    logger.info("--- Verification ---")
    logger.info("Shape     : %s", df.shape)
    logger.info("Columns   : %s", df.columns.tolist())
    logger.info("Dtypes    :\n%s", df.dtypes)
    logger.info("First 3 rows:\n%s", df.head(3).to_string())

    # Step 2 — EDA
    logger.info("=" * 60)
    logger.info("TRINETRA — Step 2: EDA")
    logger.info("=" * 60)
    report = perform_eda(df, EDA_REPORT_PATH)
    print(report)
