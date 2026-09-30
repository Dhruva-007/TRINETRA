#!/bin/bash
set -e

INPUT="data/raw/test.pcap"
OUTPUT="data/flows/packets.csv"

echo "======================================"
echo " TRINETRA - PACKET EXTRACTION"
echo "======================================"

if [ ! -f "$INPUT" ]; then
    echo "[ERROR] PCAP not found: $INPUT"
    exit 1
fi

echo "[TRINETRA] Reading: $INPUT"
echo "[TRINETRA] Writing: $OUTPUT"

tshark -r "$INPUT" \
-T fields \
-E header=y \
-E separator=, \
-E quote=d \
-E occurrence=f \
-e frame.time_epoch \
-e ip.src \
-e ip.dst \
-e tcp.srcport \
-e tcp.dstport \
-e udp.srcport \
-e udp.dstport \
-e ip.proto \
-e frame.len \
-e tcp.flags.syn \
-e tcp.flags.reset \
> "$OUTPUT"

echo ""
echo "[TRINETRA] Extraction complete."
echo "[TRINETRA] Output: $OUTPUT"
echo "[TRINETRA] Rows:"
wc -l "$OUTPUT"
