#!/bin/bash

set -e

echo ""
echo "=========================================="
echo "       TRINETRA DATA PIPELINE"
echo "=========================================="
echo ""

echo "[1/5] Extracting packets..."
./scripts/extract_packets.sh

echo ""
echo "[2/5] Building flows..."
python3 scripts/flow_extractor.py

echo ""
echo "[3/5] Building network states..."
python3 scripts/build_network_state.py

echo ""
echo "[4/5] Building temporal deltas..."
python3 scripts/build_state_deltas.py

echo ""
echo "[5/5] Building dynamic graphs..."
python3 scripts/build_graph.py

echo ""
echo "=========================================="
echo "          PIPELINE COMPLETE"
echo "=========================================="
echo ""

echo "Flow records:"
wc -l data/flows/flows.csv

echo ""
echo "Network states:"
wc -l data/windows/network_states_with_delta.csv

echo ""
echo "Dynamic graphs:"
find data/graphs \
    -name "*.json" \
    | wc -l

echo ""
echo "=========================================="
