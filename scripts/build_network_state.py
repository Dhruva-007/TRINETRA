from pathlib import Path

import numpy as np
import pandas as pd


INPUT = Path("data/flows/flows.csv")
OUTPUT = Path("data/windows/network_states.csv")

WINDOW_SECONDS = 5


def entropy(values):
    values = pd.Series(values).dropna()

    if values.empty:
        return 0.0

    probabilities = values.value_counts(
        normalize=True
    )

    return float(
        -(
            probabilities
            * np.log2(probabilities)
        ).sum()
    )


def main():

    print("[TRINETRA] Building network states...")

    if not INPUT.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT}"
        )

    df = pd.read_csv(INPUT)

    states = []

    previous_peers = set()

    for window, group in df.groupby("window"):

        sources = set(
            group["src_ip"].dropna()
        )

        destinations = set(
            group["dst_ip"].dropna()
        )

        peers = sources | destinations

        new_peers = peers - previous_peers

        total_connections = len(group)

        total_packets = group["packets"].sum()

        total_bytes = group["bytes"].sum()

        unique_sources = len(sources)

        unique_destinations = len(
            destinations
        )

        unique_peers = len(peers)

        # Number of distinct destination ports
        # observed in this complete window.
        unique_ports = (
            group["unique_dst_ports"]
            .dropna()
            .astype(int)
            .sum()
        )

        syn_packets = (
            group["syn_packets"]
            .fillna(0)
            .sum()
        )

        rst_packets = (
            group["rst_packets"]
            .fillna(0)
            .sum()
        )

        syn_ratio = (
            syn_packets / total_packets
            if total_packets > 0
            else 0.0
        )

        rst_ratio = (
            rst_packets / total_packets
            if total_packets > 0
            else 0.0
        )

        durations = (
            group["duration"]
            .fillna(0)
        )

        average_duration = (
            durations.mean()
            if not durations.empty
            else 0.0
        )

        state = {
            "window": int(window),

            "total_connections":
                int(total_connections),

            "unique_sources":
                int(unique_sources),

            "unique_destinations":
                int(unique_destinations),

            "unique_peers":
                int(unique_peers),

            "unique_ports":
                int(unique_ports),

            "bytes":
                float(total_bytes),

            "packets":
                float(total_packets),

            "connection_rate":
                float(
                    total_connections
                    / WINDOW_SECONDS
                ),

            "new_peer_rate":
                float(
                    len(new_peers)
                    / WINDOW_SECONDS
                ),

            "syn_ratio":
                float(syn_ratio),

            "rst_ratio":
                float(rst_ratio),

            "average_duration":
                float(average_duration),

            "communication_entropy":
                entropy(group["src_ip"]),
        }

        states.append(state)

        previous_peers = peers

    result = pd.DataFrame(states)

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    result.to_csv(
        OUTPUT,
        index=False
    )

    print(
        f"[TRINETRA] Saved "
        f"{len(result)} network states."
    )

    print(
        f"[TRINETRA] Output: {OUTPUT}"
    )

    print("\nNetwork states:")
    print(result)


if __name__ == "__main__":
    main()
