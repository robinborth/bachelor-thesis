from typing import Dict, List, Union

import networkx as nx
import pandas as pd
from networkx import Graph

from src.utils import load_graph, save_report


def nodes_with_one_path_to_crash(G: Graph) -> int:
    nodes = 0
    for node in nx.ancestors(G, "crash"):
        if nx.all_shortest_paths(G, source=node, target="crash"):
            nodes += 1
    return nodes


def is_subset(path: List[str], paths: List[List[str]]):
    return any((set(path).issubset(set(p)) for p in paths))


def paths_to_crash(G: Graph) -> List[List[str]]:
    paths: List[List[str]] = []
    for node in nx.ancestors(G, "crash"):
        for path in nx.all_shortest_paths(G, source=node, target="crash"):
            if not is_subset(path, paths):
                paths.append(path)
    return paths


def generate_graphs_report() -> None:
    report: List[Dict[str, Union[str, int]]] = []
    for name in ["baseline", "lemma", "pos", "root"]:
        graph = load_graph(name)
        report.append(
            {
                "name": name,
                "nodes": len(graph.nodes),
                "edges": len(graph.edges),
                "density": len(graph.edges()) / (len(graph.nodes()) * (len(graph.nodes()) - 1)),
                "nodes_with_one_path_to_crash": nodes_with_one_path_to_crash(graph),
                "total_distinct_paths_to_crash": len(paths_to_crash(graph)),
            }
        )
    text = str(pd.DataFrame(report).to_string())
    save_report(text, "graph_characteristics.txt")


if __name__ == "__main__":
    generate_graphs_report()
