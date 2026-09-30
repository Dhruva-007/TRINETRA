# TRINETRA
## Network Trajectory Intelligence & Early Warning

TRINETRA is a lightweight cybersecurity prototype for analyzing
network behaviour as an evolving sequence of network states rather
than treating individual packets or flows in isolation.

The system combines network telemetry, temporal state representation,
dynamic communication graphs, anomaly detection, and temporal
forecasting to provide security-oriented situational awareness.

---

## 1. Core Idea

Traditional network monitoring often focuses on detecting abnormal
events in the current network state.

TRINETRA additionally models how the network is changing over time.

The system therefore attempts to answer five questions:

1. WHAT is unusual?
2. WHERE is the unusual behaviour occurring?
3. WHEN did the behaviour begin changing?
4. WHY does the system consider it unusual?
5. WHERE is the current trajectory heading?

The prototype is designed as a modular pipeline so that network
telemetry, machine learning, security reasoning, and visualization
can be developed independently.

---

## 2. System Architecture

```text
              PCAP / LIVE NETWORK TRAFFIC
                         |
                         v
                  TShark / Zeek
                         |
                         v
                    Flow Records
                         |
                         v
                  Temporal Windows
                         |
                         v
                Dynamic Network State
                         |
              +----------+----------+
              |                     |
              v                     v
       Anomaly Detection      GRU Forecasting
              |                     |
              v                     v
       Current Anomaly        Future Risk
              |                     |
              +----------+----------+
                         |
                         v
                  Security Engine
                         |
              +----------+----------+
              |          |          |
              v          v          v
             WHAT       WHERE      WHY
              |
              v
          WHEN / TRAJECTORY
              |
              v
            SOC Dashboard
