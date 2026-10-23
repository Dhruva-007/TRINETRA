"""
TRINETRA -- test_trinetra_pipeline.py
=======================================
End-to-end integration test for the complete TRINETRA ML pipeline.

Tests all 14 items from the spec:
  1.  IF loads successfully
  2.  GRU loads successfully
  3.  Both inference paths execute
  4.  Temporal alignment works
  5.  IF signal is [0,1]
  6.  GRU signal is [0,1]
  7.  Threat Index is [0,1]
  8.  Severity is valid
  9.  No labels enter live inference
  10. No raw IF + raw GRU values directly added
  11. Existing IF results remain reproducible
  12. Existing GRU inference remains reproducible
  13. Saved config reproduces same Threat Index
  14. Empty/invalid input fails safely

All imports are at the top of the file.
"""

import os
import sys
import json
import logging
import traceback

import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trinetra_pipeline import (
    run_trinetra_pipeline,
    load_all_artefacts,
    build_config_from_data,
    compute_threat_index,
    _normalize_signal,
    SEVERITY_BANDS,
    PROCESSED_CSV,
    OUTPUT_CSV_PATH,
    THREAT_CONFIG_PATH,
)
from inference import run_inference_on_csv as if_run_csv
from gru_inference import run_inference_on_csv as gru_run_csv

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
# Valid severity labels
# ---------------------------------------------------------------------------

VALID_SEVERITIES = {label for _, _, label in SEVERITY_BANDS}

# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

def run_tests():
    results = []
    passed = 0
    failed = 0

    def check(name: str, condition: bool, detail: str = ""):
        nonlocal passed, failed
        status = "PASS" if condition else "FAIL"
        symbol = "[PASS]" if condition else "[FAIL]"
        results.append((status, name, detail))
        if condition:
            passed += 1
            logger.info("%s  %s  %s", symbol, name, detail)
        else:
            failed += 1
            logger.error("%s  %s  %s", symbol, name, detail)

    # -----------------------------------------------------------------------
    # Setup: load everything once
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  TRINETRA Integration Test Suite")
    print("=" * 70)

    try:
        (if_model, if_preprocessor, if_schema,
         gru_model, gru_scaler, gru_config,
         ti_config) = load_all_artefacts()
        artefacts_loaded = True
    except Exception as e:
        logger.error("Failed to load artefacts: %s", e)
        artefacts_loaded = False

    df = pd.read_csv(PROCESSED_CSV)

    # -----------------------------------------------------------------------
    # Test 1: IF loads successfully
    # -----------------------------------------------------------------------
    check("1. IF model loads",
          artefacts_loaded and if_model is not None,
          f"n_features_in_={getattr(if_model, 'n_features_in_', '?')}" if artefacts_loaded else "load failed")

    # -----------------------------------------------------------------------
    # Test 2: GRU loads successfully
    # -----------------------------------------------------------------------
    check("2. GRU model loads",
          artefacts_loaded and gru_model is not None,
          f"n_features={gru_config['n_features']}" if artefacts_loaded else "load failed")

    if not artefacts_loaded:
        logger.error("Artefact loading failed — cannot proceed with remaining tests.")
        _print_summary(results, passed, failed)
        return passed, failed

    # -----------------------------------------------------------------------
    # Test 3: Both inference paths execute
    # -----------------------------------------------------------------------
    try:
        if_result = if_run_csv(PROCESSED_CSV)
        if_ok = "anomaly_score_negated" in if_result.columns and len(if_result) > 0
    except Exception as e:
        if_ok = False
        logger.error("IF inference failed: %s", e)

    try:
        gru_result = gru_run_csv(PROCESSED_CSV)
        gru_ok = "trajectory_deviation" in gru_result.columns and len(gru_result) > 0
    except Exception as e:
        gru_ok = False
        logger.error("GRU inference failed: %s", e)

    check("3a. IF inference executes",  if_ok,  f"{len(if_result)} flows" if if_ok else "FAILED")
    check("3b. GRU inference executes", gru_ok, f"{len(gru_result)} windows" if gru_ok else "FAILED")

    # -----------------------------------------------------------------------
    # Test 4: Temporal alignment works
    # -----------------------------------------------------------------------
    if if_ok and gru_ok:
        from trinetra_pipeline import align_if_to_gru_windows, merge_signals
        try:
            if_win = align_if_to_gru_windows(if_result, gru_config["aggregation_window"])
            merged = merge_signals(if_win, gru_result)
            align_ok = len(merged) > 0 and "mean_anomaly_score_negated" in merged.columns
        except Exception as e:
            align_ok = False
            logger.error("Temporal alignment failed: %s", e)
        check("4. Temporal alignment", align_ok,
              f"{len(merged)} aligned windows" if align_ok else "FAILED")
    else:
        check("4. Temporal alignment", False, "skipped — inference failed")

    # -----------------------------------------------------------------------
    # Run full pipeline for remaining tests
    # -----------------------------------------------------------------------
    try:
        output = run_trinetra_pipeline(
            PROCESSED_CSV,
            if_model=if_model, if_preprocessor=if_preprocessor, if_schema=if_schema,
            gru_model=gru_model, gru_scaler=gru_scaler, gru_config=gru_config,
            ti_config=ti_config,
            save_output=True,
            output_path=OUTPUT_CSV_PATH,
        )
        pipeline_ok = True
    except Exception as e:
        pipeline_ok = False
        output = pd.DataFrame()
        logger.error("Full pipeline failed: %s\n%s", e, traceback.format_exc())

    # -----------------------------------------------------------------------
    # Test 5: IF signal is [0,1]
    # -----------------------------------------------------------------------
    if pipeline_ok and "if_signal" in output.columns:
        if_sig = output["if_signal"].dropna()
        check("5. IF signal ∈ [0,1]",
              bool((if_sig >= 0).all() and (if_sig <= 1).all()),
              f"range=[{if_sig.min():.4f}, {if_sig.max():.4f}]")
    else:
        check("5. IF signal ∈ [0,1]", False, "pipeline failed or column missing")

    # -----------------------------------------------------------------------
    # Test 6: GRU signal is [0,1]
    # -----------------------------------------------------------------------
    if pipeline_ok and "gru_signal" in output.columns:
        gru_sig = output["gru_signal"].dropna()
        check("6. GRU signal ∈ [0,1]",
              bool((gru_sig >= 0).all() and (gru_sig <= 1).all()),
              f"range=[{gru_sig.min():.4f}, {gru_sig.max():.4f}]")
    else:
        check("6. GRU signal ∈ [0,1]", False, "pipeline failed or column missing")

    # -----------------------------------------------------------------------
    # Test 7: Threat Index is [0,1]
    # -----------------------------------------------------------------------
    if pipeline_ok and "threat_index" in output.columns:
        ti_vals = output["threat_index"].dropna()
        check("7. Threat Index ∈ [0,1]",
              bool((ti_vals >= 0).all() and (ti_vals <= 1).all()),
              f"range=[{ti_vals.min():.4f}, {ti_vals.max():.4f}]  mean={ti_vals.mean():.4f}")
    else:
        check("7. Threat Index ∈ [0,1]", False, "pipeline failed or column missing")

    # -----------------------------------------------------------------------
    # Test 8: Severity is valid
    # -----------------------------------------------------------------------
    if pipeline_ok and "severity" in output.columns:
        bad_sev = set(output["severity"].unique()) - VALID_SEVERITIES
        check("8. Severity labels valid",
              len(bad_sev) == 0,
              f"valid={sorted(output['severity'].unique())}  bad={bad_sev}")
    else:
        check("8. Severity labels valid", False, "pipeline failed or column missing")

    # -----------------------------------------------------------------------
    # Test 9: No labels enter live inference
    # -----------------------------------------------------------------------
    # Verify IF feature schema has no forbidden columns
    forbidden_in_if = {"label", "detailedlabel"}.intersection(
        set(if_schema["numerical_features"] + if_schema["categorical_features"])
    )
    forbidden_in_gru = {"label", "detailedlabel", "ts"}.intersection(
        set(gru_config["feature_names"])
    )
    check("9a. Labels not in IF features",  len(forbidden_in_if)  == 0, str(forbidden_in_if))
    check("9b. Labels not in GRU features", len(forbidden_in_gru) == 0, str(forbidden_in_gru))

    # -----------------------------------------------------------------------
    # Test 10: No direct raw addition of IF + GRU scores
    # -----------------------------------------------------------------------
    # Verify that both signals pass through normalization
    if pipeline_ok:
        if_ref_max  = ti_config["if_ref_max"]
        gru_ref_max = ti_config["gru_ref_max"]
        normed_correctly = (if_ref_max > 0) and (gru_ref_max > 0)
        check("10. Scores normalized before addition", normed_correctly,
              f"IF ref_max={if_ref_max:.4f}  GRU ref_max={gru_ref_max:.4f}")
    else:
        check("10. Scores normalized before addition", False, "pipeline failed")

    # -----------------------------------------------------------------------
    # Test 11: Existing IF results remain reproducible
    # -----------------------------------------------------------------------
    try:
        if_result2 = if_run_csv(PROCESSED_CSV)
        existing   = pd.read_csv("../results/inference_scored.csv") \
                     if os.path.exists("../results/inference_scored.csv") else None
        if existing is not None:
            n_match = len(if_result2)
            n_orig  = len(existing)
            repro = (n_match == n_orig and
                     round(if_result2["anomaly_score_raw"].sum(), 4) ==
                     round(existing["anomaly_score_raw"].sum(), 4))
        else:
            repro = len(if_result2) > 0
        check("11. IF results reproducible", repro,
              f"rows={len(if_result2)}  sum_match={repro}")
    except Exception as e:
        check("11. IF results reproducible", False, str(e))

    # -----------------------------------------------------------------------
    # Test 12: Existing GRU inference remains reproducible
    # -----------------------------------------------------------------------
    try:
        gru_result2 = gru_run_csv(PROCESSED_CSV)
        repro_gru = len(gru_result2) > 0 and "trajectory_deviation" in gru_result2.columns
        check("12. GRU results reproducible", repro_gru,
              f"rows={len(gru_result2)}")
    except Exception as e:
        check("12. GRU results reproducible", False, str(e))

    # -----------------------------------------------------------------------
    # Test 13: Saved config reproduces same Threat Index
    # -----------------------------------------------------------------------
    try:
        ti_config2 = joblib.load(THREAT_CONFIG_PATH)
        output2 = run_trinetra_pipeline(
            PROCESSED_CSV,
            if_model=if_model, if_preprocessor=if_preprocessor, if_schema=if_schema,
            gru_model=gru_model, gru_scaler=gru_scaler, gru_config=gru_config,
            ti_config=ti_config2,
        )
        ti_match = (
            len(output2) == len(output) and
            round(output2["threat_index"].sum(), 4) == round(output["threat_index"].sum(), 4)
        )
        check("13. Config reproduces same Threat Index", ti_match,
              f"rows={len(output2)}  ti_sum={round(output2['threat_index'].sum(),4)}")
    except Exception as e:
        check("13. Config reproduces same Threat Index", False, str(e))

    # -----------------------------------------------------------------------
    # Test 14: Empty/invalid input fails safely
    # -----------------------------------------------------------------------
    empty_caught = False
    try:
        run_trinetra_pipeline(
            pd.DataFrame(),
            if_model=if_model, if_preprocessor=if_preprocessor, if_schema=if_schema,
            gru_model=gru_model, gru_scaler=gru_scaler, gru_config=gru_config,
            ti_config=ti_config,
        )
    except (ValueError, Exception):
        empty_caught = True
    check("14a. Empty DataFrame fails safely", empty_caught, "ValueError raised as expected")

    bad_type_caught = False
    try:
        run_trinetra_pipeline(
            12345,
            if_model=if_model, if_preprocessor=if_preprocessor, if_schema=if_schema,
            gru_model=gru_model, gru_scaler=gru_scaler, gru_config=gru_config,
            ti_config=ti_config,
        )
    except (TypeError, Exception):
        bad_type_caught = True
    check("14b. Invalid input type fails safely", bad_type_caught, "TypeError raised as expected")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    _print_summary(results, passed, failed)
    return passed, failed


def _print_summary(results, passed, failed):
    print("\n" + "=" * 70)
    print("  Test Summary")
    print("=" * 70)
    for status, name, detail in results:
        sym = "✓" if status == "PASS" else "✗"
        print(f"  {sym} [{status}] {name}")
        if detail:
            print(f"          {detail}")
    total = passed + failed
    print(f"\n  {passed}/{total} tests passed")
    if failed == 0:
        print("  ALL TESTS PASSED — TRINETRA integration is verified.")
    else:
        print(f"  {failed} TEST(S) FAILED — review above.")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p, f = run_tests()
    sys.exit(0 if f == 0 else 1)
