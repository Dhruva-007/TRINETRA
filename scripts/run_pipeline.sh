#!/bin/bash
set -e

echo ""
echo "=========================================="
echo "       TRINETRA TELEMETRY PIPELINE"
echo "=========================================="
echo ""

PCAP="data/raw/test.pcap"
ZEEK_OUTPUT="data/zeek"

if [ ! -f "$PCAP" ]; then
    echo "[ERROR] PCAP not found: $PCAP"
    exit 1
fi

mkdir -p "$ZEEK_OUTPUT"

echo "[1/6] Running Zeek..."
rm -f "$ZEEK_OUTPUT"/*.log
(
    cd "$ZEEK_OUTPUT"
    /opt/zeek/bin/zeek -r "../../$PCAP"
)

echo "[2/6] Extracting packets with TShark..."
./scripts/extract_packets.sh

echo "[3/6] Building flows..."
python scripts/flow_extractor.py

echo "[4/6] Building network states..."
python scripts/build_network_state.py

echo "[5/6] Building temporal deltas..."
python scripts/build_state_deltas.py

echo "[6/6] Building dynamic graphs..."
python scripts/build_graph.py

echo ""
echo "=========================================="
echo "       TELEMETRY PIPELINE COMPLETE"
echo "=========================================="
echo ""

echo "Zeek:"
wc -l "$ZEEK_OUTPUT/conn.log"

echo ""
echo "Packets:"
wc -l data/flows/packets.csv

echo ""
echo "Flows:"
wc -l data/flows/flows.csv

echo ""
echo "Network states:"
wc -l data/windows/network_states_with_delta.csv

echo ""
echo "Graphs:"
find data/graphs -name "*.json" | wc -l

echo ""
echo "=========================================="
