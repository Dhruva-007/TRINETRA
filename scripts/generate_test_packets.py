from pathlib import Path
import random
import pandas as pd


OUTPUT = Path("data/flows/packets.csv")

random.seed(42)


def add_packet(
    rows,
    timestamp,
    src,
    dst,
    src_port,
    dst_port,
    protocol=6,
    length=500,
    syn=0,
    rst=0,
):
    rows.append({
        "frame.time_epoch": timestamp,
        "ip.src": src,
        "ip.dst": dst,
        "tcp.srcport": src_port if protocol == 6 else None,
        "tcp.dstport": dst_port if protocol == 6 else None,
        "udp.srcport": src_port if protocol == 17 else None,
        "udp.dstport": dst_port if protocol == 17 else None,
        "ip.proto": protocol,
        "frame.len": length,
        "tcp.flags.syn": syn,
        "tcp.flags.reset": rst,
    })


def generate_normal_traffic(rows, start_time):
    """
    Generate ordinary communication between a small
    number of known hosts.
    """

    hosts = [
        "10.0.0.10",
        "10.0.0.20",
        "10.0.0.30",
    ]

    servers = [
        ("10.0.0.100", 80),
        ("10.0.0.101", 443),
    ]

    timestamp = start_time

    for _ in range(30):

        src = random.choice(hosts)
        dst, dst_port = random.choice(servers)

        add_packet(
            rows,
            timestamp,
            src,
            dst,
            random.randint(40000, 50000),
            dst_port,
            protocol=6,
            length=random.randint(400, 1200),
            syn=1,
        )

        timestamp += random.uniform(
            0.05,
            0.20,
        )

        # Response traffic
        add_packet(
            rows,
            timestamp,
            dst,
            src,
            dst_port,
            random.randint(40000, 50000),
            protocol=6,
            length=random.randint(500, 1400),
        )

        timestamp += random.uniform(
            0.05,
            0.20,
        )

    return timestamp


def generate_suspicious_traffic(rows, start_time):
    """
    Generate a controlled behavioural shift:
    - new peer
    - many new destination ports
    - connection spike
    - SYN-heavy traffic
    - RST responses
    """

    timestamp = start_time

    attacker = "10.0.0.50"
    target = "10.0.0.200"

    # New peer + port sweep-like behaviour
    for port in range(20, 70):

        add_packet(
            rows,
            timestamp,
            attacker,
            target,
            random.randint(40000, 60000),
            port,
            protocol=6,
            length=60,
            syn=1,
        )

        timestamp += 0.03

        # Some ports respond with RST
        if port % 3 == 0:

            add_packet(
                rows,
                timestamp,
                target,
                attacker,
                port,
                random.randint(40000, 60000),
                protocol=6,
                length=60,
                rst=1,
            )

            timestamp += 0.02

    # Additional connection burst
    for _ in range(40):

        dst = random.choice([
            "10.0.0.201",
            "10.0.0.202",
            "10.0.0.203",
            "10.0.0.204",
        ])

        add_packet(
            rows,
            timestamp,
            attacker,
            dst,
            random.randint(40000, 60000),
            random.randint(1, 1024),
            protocol=6,
            length=60,
            syn=1,
        )

        timestamp += 0.02

    return timestamp


def main():

    rows = []

    timestamp = 1727000000.0

    print("[TRINETRA] Generating synthetic traffic...")

    # Windows 0-5 approximately:
    # normal traffic
    timestamp = generate_normal_traffic(
        rows,
        timestamp,
    )

    # Suspicious behavioural transition
    timestamp += 1.0

    generate_suspicious_traffic(
        rows,
        timestamp,
    )

    df = pd.DataFrame(rows)

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT,
        index=False,
    )

    print(
        f"[TRINETRA] Generated {len(df)} packets."
    )

    print(
        f"[TRINETRA] Saved to {OUTPUT}"
    )

    print("\nPacket preview:")
    print(df.head())

    print("\nUnique sources:")
    print(df["ip.src"].unique())

    print("\nUnique destinations:")
    print(df["ip.dst"].unique())


if __name__ == "__main__":
    main()
