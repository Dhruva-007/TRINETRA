from scapy.all import IP, TCP, wrpcap
from pathlib import Path
import random
import time

OUTPUT = Path("data/raw/test.pcap")
random.seed(42)

packets = []

base_time = time.time()

# Normal traffic
hosts = ["10.0.0.10", "10.0.0.20", "10.0.0.30"]
servers = [
    ("10.0.0.100", 80),
    ("10.0.0.101", 443),
]

t = base_time

for _ in range(40):
    src = random.choice(hosts)
    dst, port = random.choice(servers)
    sport = random.randint(40000, 50000)

    syn = IP(src=src, dst=dst) / TCP(
        sport=sport,
        dport=port,
        flags="S"
    )
    syn.time = t
    packets.append(syn)

    ack = IP(src=src, dst=dst) / TCP(
        sport=sport,
        dport=port,
        flags="A"
    )
    ack.time = t + 0.1
    packets.append(ack)

    t += random.uniform(0.2, 0.5)


# Suspicious port-scanning-like traffic
attacker = "10.0.0.50"
target = "10.0.0.200"

for port in range(20, 70):
    sport = random.randint(40000, 60000)

    syn = IP(src=attacker, dst=target) / TCP(
        sport=sport,
        dport=port,
        flags="S"
    )
    syn.time = t
    packets.append(syn)

    if port % 3 == 0:
        rst = IP(src=target, dst=attacker) / TCP(
            sport=port,
            dport=sport,
            flags="R"
        )
        rst.time = t + 0.03
        packets.append(rst)

    t += 0.05


# Additional burst to multiple hosts
for dst in [
    "10.0.0.201",
    "10.0.0.202",
    "10.0.0.203",
    "10.0.0.204",
]:
    for port in range(1, 20):
        sport = random.randint(40000, 60000)

        pkt = IP(src=attacker, dst=dst) / TCP(
            sport=sport,
            dport=port,
            flags="S"
        )
        pkt.time = t
        packets.append(pkt)

        t += 0.03


OUTPUT.parent.mkdir(parents=True, exist_ok=True)
wrpcap(str(OUTPUT), packets)

print(f"[TRINETRA] Generated {len(packets)} packets")
print(f"[TRINETRA] PCAP: {OUTPUT}")
