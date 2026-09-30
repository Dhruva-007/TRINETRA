from pathlib import Path

import pandas as pd


INPUT = Path("data/flows/packets.csv")
OUTPUT = Path("data/flows/flows.csv")

WINDOW_SECONDS = 5


def main():
    print("[TRINETRA] Loading packet data...")

    if not INPUT.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT}")

    df = pd.read_csv(INPUT)

    numeric_columns = [
        "frame.time_epoch",
        "frame.len",
        "ip.proto",
        "tcp.srcport",
        "tcp.dstport",
        "udp.srcport",
        "udp.dstport",
        "tcp.flags.syn",
        "tcp.flags.reset",
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(
        subset=[
            "ip.src",
            "ip.dst",
            "frame.time_epoch",
        ]
    )

    if df.empty:
        raise ValueError("No valid IP packets found.")

    # TCP or UDP source/destination port
    df["src_port"] = df["tcp.srcport"].fillna(
        df["udp.srcport"]
    )

    df["dst_port"] = df["tcp.dstport"].fillna(
        df["udp.dstport"]
    )

    # Relative time from beginning of capture
    first_timestamp = df["frame.time_epoch"].min()

    df["relative_time"] = (
        df["frame.time_epoch"] - first_timestamp
    )

    # 5-second temporal window
    df["window"] = (
        df["relative_time"] // WINDOW_SECONDS
    ).astype(int)

    group_columns = [
        "window",
        "ip.src",
        "ip.dst",
        "ip.proto",
    ]

    flows = (
        df.groupby(
            group_columns,
            dropna=False
        )
        .agg(
            packets=("frame.len", "count"),
            bytes=("frame.len", "sum"),
            first_timestamp=(
                "frame.time_epoch",
                "min"
            ),
            last_timestamp=(
                "frame.time_epoch",
                "max"
            ),
            unique_src_ports=(
                "src_port",
                "nunique"
            ),
            unique_dst_ports=(
                "dst_port",
                "nunique"
            ),
            syn_packets=(
                "tcp.flags.syn",
                "sum"
            ),
            rst_packets=(
                "tcp.flags.reset",
                "sum"
            ),
        )
        .reset_index()
    )

    flows["duration"] = (
        flows["last_timestamp"]
        - flows["first_timestamp"]
    )

    flows = flows.rename(
        columns={
            "ip.src": "src_ip",
            "ip.dst": "dst_ip",
            "ip.proto": "protocol",
        }
    )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    flows.to_csv(
        OUTPUT,
        index=False
    )

    print(
        f"[TRINETRA] Saved {len(flows)} flow records."
    )

    print(
        f"[TRINETRA] Output: {OUTPUT}"
    )

    print("\nFirst records:")
    print(flows.head())


if __name__ == "__main__":
    main()
