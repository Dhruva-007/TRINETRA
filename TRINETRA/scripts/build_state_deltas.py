import pandas as pd


INPUT = "data/windows/network_states.csv"
OUTPUT = "data/windows/network_states_with_delta.csv"


FEATURES = [
    "total_connections",
    "unique_sources",
    "unique_destinations",
    "unique_peers",
    "unique_ports",
    "bytes",
    "packets",
    "connection_rate",
    "new_peer_rate",
    "syn_ratio",
    "rst_ratio",
    "average_duration",
    "communication_entropy",
]


def main():

    print("[TRINETRA] Building temporal deltas...")

    df = pd.read_csv(INPUT)

    for feature in FEATURES:

        df[f"{feature}_delta"] = (
            df[feature].diff()
        )

    # First window has no previous state.
    df = df.fillna(0)

    df.to_csv(
        OUTPUT,
        index=False
    )

    print(
        f"[TRINETRA] Saved: {OUTPUT}"
    )

    print(
        f"[TRINETRA] Features: {len(FEATURES)}"
    )

    print(
        f"[TRINETRA] Total columns: "
        f"{len(df.columns)}"
    )

    print("\nPreview:")
    print(df.head())


if __name__ == "__main__":
    main()
