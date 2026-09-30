# TRINETRA Team Handoff

## Member 1 — ML Engine

Primary input:

`network_states_with_delta.csv`

Location:

`handoff/member1/`

Each row represents one 5-second network state.

Use:

- Base features for current network state
- Delta features for temporal trajectory

Optional:

- Dynamic graph JSON files

---

## Member 3 — Dashboard

Primary inputs:

`network_states_with_delta.csv`

`graphs/*.json`

Security events will eventually be provided through:

`data/events/`

Dashboard should not directly depend on the packet-level PCAP.

---

## Member 2 — Network Telemetry

Responsible for:

PCAP / traffic
→ packet extraction
→ flow aggregation
→ temporal windows
→ network-state construction
→ temporal deltas
→ dynamic graphs
