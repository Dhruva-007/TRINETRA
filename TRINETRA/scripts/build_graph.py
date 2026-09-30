import json
from pathlib import Path

import networkx as nx
import pandas as pd


INPUT = Path("data/flows/flows.csv")
OUTPUT = Path("data/graphs")


def main():

    print("[TRINETRA] Building dynamic graphs...")

    if not INPUT.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT}"
        )

    df = pd.read_csv(INPUT)

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True
    )

    graph_count = 0

    for window, group in df.groupby("window"):

        graph = nx.DiGraph()

        for _, row in group.iterrows():

            source = str(row["src_ip"])
            destination = str(row["dst_ip"])

            packets = float(
                row["packets"]
            )

            bytes_value = float(
                row["bytes"]
            )

            if graph.has_edge(
                source,
                destination
            ):

                edge = graph[
                    source
                ][
                    destination
                ]

                edge["packets"] += packets

                edge["bytes"] += bytes_value

            else:

                graph.add_edge(
                    source,
                    destination,
                    packets=packets,
                    bytes=bytes_value,
                    protocol=str(
                        row["protocol"]
                    ),
                )

        # Structural node features
        for node in graph.nodes():

            graph.nodes[node][
                "in_degree"
            ] = graph.in_degree(node)

            graph.nodes[node][
                "out_degree"
            ] = graph.out_degree(node)

            graph.nodes[node][
                "degree"
            ] = graph.degree(node)

        output_file = (
            OUTPUT
            / f"graph_{int(window):04d}.json"
        )

        data = nx.node_link_data(graph, edges="links")

        with open(
            output_file,
            "w"
        ) as f:

            json.dump(
                data,
                f,
                indent=2
            )

        graph_count += 1

    print(
        f"[TRINETRA] Generated "
        f"{graph_count} dynamic graphs."
    )


if __name__ == "__main__":
    main()
