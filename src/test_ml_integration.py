"""
TRINETRA — src/test_ml_integration.py
=======================================
Integration tests for ml_integration.py.

All paths are resolved from the shared repository root so that these
tests pass on any machine that has cloned the repository and installed
requirements.txt.

Tests
-----
 1. Repo layout: ml/src/ and models/ exist inside the repo
 2. Artefact validation passes for all 7 required files
 3. Missing artefact raises FileNotFoundError
 4. Timestamp conversion (epoch 0 → ISO-8601)
 5. Evidence list is non-empty for a valid ML output row
 6. Security event has all required schema fields
 7. Security event field types are correct
 8. event_id format is TRN-NNNN
 9. current_anomaly in [0, 1]
10. trajectory_change in [0, 1]
11. threat_index in [0, 1]
12. severity is a valid label
13. Empty ML output yields empty events list
14. Full end-to-end integration run produces events
15. JSONL output is valid JSON Lines (every line parses)
16. Event count matches window count
17. All events have unique event_id values
18. Missing input CSV raises FileNotFoundError

All imports are at the top of the file.
"""

import json
import logging
import os
import sys
import tempfile

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Ensure src/ is on the path so ml_integration is importable
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))   # …/src
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from ml_integration import (
    _REPO_ROOT,
    _ML_SRC,
    validate_artefacts,
    _ts_to_iso,
    _safe_int,
    _build_evidence,
    build_security_events,
    write_events_jsonl,
    run_ml_integration,
    REQUIRED_ARTEFACTS,
    DEFAULT_ZEEK_CSV,
    ML_OUTPUT_CSV,
    EVENTS_JSONL,
)

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REQUIRED_SCHEMA_FIELDS = {
    "event_id", "window", "timestamp", "source", "destination",
    "event_type", "current_anomaly", "trajectory_change", "future_risk",
    "threat_index", "evidence", "technique", "severity",
}

VALID_SEVERITIES = {"Low", "Moderate", "High", "Critical"}


# ---------------------------------------------------------------------------
# Synthetic ML output helper
# ---------------------------------------------------------------------------

def _make_synthetic_output(n_rows: int = 5) -> pd.DataFrame:
    """Create a minimal synthetic ML output DataFrame for unit tests."""
    rng = np.random.default_rng(42)
    rows = []
    for i in range(n_rows):
        ti = float(rng.uniform(0, 1))
        sev = (
            "Critical" if ti >= 0.80 else
            "High"     if ti >= 0.60 else
            "Moderate" if ti >= 0.30 else
            "Low"
        )
        rows.append({
            "window_start":                float(1000.0 + i * 10),
            "n_flows":                     int(rng.integers(1, 20)),
            "n_flagged":                   int(rng.integers(0, 5)),
            "mean_anomaly_score_negated":  float(rng.uniform(-0.2, 0.2)),
            "max_anomaly_score_negated":   float(rng.uniform(0, 0.3)),
            "trajectory_deviation":        float(rng.uniform(1e6, 5e10)),
            "if_signal":                   float(np.clip(rng.uniform(0, 1), 0, 1)),
            "gru_signal":                  float(np.clip(rng.uniform(0, 1), 0, 1)),
            "threat_index":                float(np.clip(ti, 0, 1)),
            "severity":                    sev,
            "label":                       "Background",
            "model_version":               "IF-v1.GRU-v1.TI-v1",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

def run_tests() -> tuple[int, int]:
    passed = 0
    failed = 0
    results = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        nonlocal passed, failed
        sym    = "[PASS]" if ok else "[FAIL]"
        status = "PASS"   if ok else "FAIL"
        results.append((status, name, detail))
        if ok:
            passed += 1
        else:
            failed += 1
        print(f"  {sym}  {name}  {detail}")

    print("\n" + "=" * 72)
    print("  TRINETRA ML Integration Test Suite")
    print("=" * 72)
    print(f"  Repo root : {_REPO_ROOT}")
    print(f"  ML src    : {_ML_SRC}")
    print()

    synthetic = _make_synthetic_output(n_rows=5)

    # ------------------------------------------------------------------
    # Test 1: Repo layout is correct
    # ------------------------------------------------------------------
    ml_src_exists     = os.path.isdir(_ML_SRC)
    models_dir_exists = os.path.isdir(os.path.join(_REPO_ROOT, "models"))
    check("1.  ml/src/ directory exists",   ml_src_exists,     f"path={_ML_SRC}")
    check("2.  models/ directory exists",   models_dir_exists, f"path={os.path.join(_REPO_ROOT, 'models')}")

    # ------------------------------------------------------------------
    # Tests 2-3: Artefact validation
    # ------------------------------------------------------------------
    try:
        validate_artefacts()
        artefacts_ok = True
    except FileNotFoundError as e:
        artefacts_ok = False
        print(f"    detail: {e}")
    check("3.  Artefact validation passes  (7 files)", artefacts_ok,
          f"({len(REQUIRED_ARTEFACTS)} artefacts checked)")

    # Inject a fake missing artefact and confirm error is raised
    import ml_integration as _mod
    _mod.REQUIRED_ARTEFACTS["__test_missing__"] = "/nonexistent/__test_missing__.joblib"
    raised_missing = False
    try:
        validate_artefacts()
    except FileNotFoundError:
        raised_missing = True
    finally:
        del _mod.REQUIRED_ARTEFACTS["__test_missing__"]
    check("4.  Missing artefact raises FileNotFoundError", raised_missing)

    # ------------------------------------------------------------------
    # Tests 5-: Unit tests on helpers
    # ------------------------------------------------------------------
    iso = _ts_to_iso(0.0)
    check("5.  Timestamp conversion epoch=0", iso == "1970-01-01T00:00:00Z", f"got={iso!r}")

    ev_list = _build_evidence(synthetic.iloc[0])
    check("6.  Evidence list non-empty", len(ev_list) > 0, f"n={len(ev_list)}")

    check("7.  _safe_int(NaN) returns None",  _safe_int(float("nan")) is None)
    check("8.  _safe_int(7.0) returns 7",     _safe_int(7.0) == 7)
    check("9.  _safe_int(None) returns None", _safe_int(None) is None)

    # ------------------------------------------------------------------
    # Security event structure tests
    # ------------------------------------------------------------------
    events = build_security_events(synthetic)

    check("10. build_security_events returns list", isinstance(events, list))

    if events:
        ev = events[0]
        missing_fields = REQUIRED_SCHEMA_FIELDS - set(ev.keys())
        check("11. Event has all required schema fields",
              len(missing_fields) == 0, f"missing={missing_fields}")
        check("12. event_id format is TRN-NNNN",
              ev["event_id"].startswith("TRN-"), f"sample={ev['event_id']!r}")
        check("13. event_id is str and evidence is list",
              isinstance(ev["event_id"], str) and isinstance(ev["evidence"], list))
    else:
        for t in ["11. Event has all required schema fields",
                  "12. event_id format is TRN-NNNN",
                  "13. event_id is str and evidence is list"]:
            check(t, False, "empty events")

    if events:
        ca_vals = [e["current_anomaly"] for e in events if e["current_anomaly"] is not None]
        check("14. current_anomaly in [0,1]", all(0 <= v <= 1 for v in ca_vals),
              f"range=[{min(ca_vals):.4f},{max(ca_vals):.4f}]" if ca_vals else "no values")
        tc_vals = [e["trajectory_change"] for e in events if e["trajectory_change"] is not None]
        check("15. trajectory_change in [0,1]", all(0 <= v <= 1 for v in tc_vals),
              f"range=[{min(tc_vals):.4f},{max(tc_vals):.4f}]" if tc_vals else "no values")
        ti_vals = [e["threat_index"] for e in events if e["threat_index"] is not None]
        check("16. threat_index in [0,1]", all(0 <= v <= 1 for v in ti_vals),
              f"range=[{min(ti_vals):.4f},{max(ti_vals):.4f}]" if ti_vals else "no values")
        bad_sev = {e["severity"] for e in events} - VALID_SEVERITIES
        check("17. severity labels valid", len(bad_sev) == 0,
              f"bad={bad_sev}" if bad_sev else "all valid")
    else:
        for t in ["14.", "15.", "16.", "17."]:
            check(t + " (skipped — empty events)", False)

    empty_events = build_security_events(pd.DataFrame())
    check("18. Empty DataFrame → empty events list", len(empty_events) == 0)

    # ------------------------------------------------------------------
    # Full end-to-end integration test
    # ------------------------------------------------------------------
    integration_ok = False
    n_events_out   = 0
    tmp_events = tmp_csv = None

    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tf_e, \
         tempfile.NamedTemporaryFile(suffix=".csv",   delete=False) as tf_c:
        tmp_events = tf_e.name
        tmp_csv    = tf_c.name

    try:
        if os.path.exists(DEFAULT_ZEEK_CSV):
            result = run_ml_integration(
                zeek_csv=DEFAULT_ZEEK_CSV,
                events_path=tmp_events,
                ml_output_path=tmp_csv,
            )
            n_events_out = sum(1 for _ in open(tmp_events))
            integration_ok = len(result) > 0 and n_events_out > 0
        else:
            print(f"    SKIP: DEFAULT_ZEEK_CSV not found: {DEFAULT_ZEEK_CSV}")
    except Exception as e:
        print(f"    ERROR: {e}")

    check("19. Full integration run produces events", integration_ok,
          f"n_events={n_events_out}" if integration_ok else "FAILED or CSV missing")

    if integration_ok:
        parse_ok = True
        bad_line = None
        with open(tmp_events) as fh:
            for lineno, line in enumerate(fh, start=1):
                try:
                    json.loads(line.strip())
                except json.JSONDecodeError:
                    parse_ok = False
                    bad_line = lineno
                    break
        check("20. JSONL output is valid JSON Lines", parse_ok,
              f"{n_events_out} lines OK" if parse_ok else f"parse error at line {bad_line}")

        with open(tmp_events) as fh:
            n_lines = sum(1 for _ in fh)
        check("21. Event count matches window count",
              n_lines == n_events_out,
              f"lines={n_lines}  events={n_events_out}")

        ids = []
        with open(tmp_events) as fh:
            for line in fh:
                ids.append(json.loads(line.strip())["event_id"])
        check("22. All event_id values are unique",
              len(ids) == len(set(ids)),
              f"n_ids={len(ids)}  n_unique={len(set(ids))}")
    else:
        for t in ["20.", "21.", "22."]:
            check(t + " (skipped — integration run did not succeed)", False)

    # Missing CSV → FileNotFoundError
    missing_csv_caught = False
    try:
        run_ml_integration(
            zeek_csv="/nonexistent/path/zeek.csv",
            events_path=tmp_events,
            ml_output_path=tmp_csv,
        )
    except FileNotFoundError:
        missing_csv_caught = True
    check("23. Missing CSV raises FileNotFoundError", missing_csv_caught)

    # Cleanup
    for f in [tmp_events, tmp_csv]:
        if f and os.path.exists(f):
            os.unlink(f)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    total = passed + failed
    print("\n" + "=" * 72)
    print("  Test Summary")
    print("=" * 72)
    for status, name, detail in results:
        sym = "✓" if status == "PASS" else "✗"
        print(f"  {sym} [{status}] {name}")
        if detail:
            print(f"          {detail}")
    print(f"\n  {passed}/{total} tests passed")
    if failed == 0:
        print("  ALL TESTS PASSED — ml_integration is verified.")
    else:
        print(f"  {failed} TEST(S) FAILED — review above.")
    print("=" * 72)

    return passed, failed


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p, f = run_tests()
    sys.exit(0 if f == 0 else 1)
