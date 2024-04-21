from typing import Any, Callable

import pandas as pd

from src.pipeline.extract import PHRASE_REQUIRED_POS
from src.utils import CauseEffectPair, load_cause_effect_pairs, save_data
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English

def reversed_weight(row: Any, graph: pd.DataFrame) -> int:
    rule = (graph["cause"] == row["effect"]) & (graph["effect"] == row["cause"])
    return int(graph.loc[rule, "weight"].sum())


def create_graph(
    data: pd.DataFrame,
    fn: Callable[[CauseEffectPair, str], str],
    weight_threshold: int,
) -> pd.DataFrame:
    """Creats a causal graph dataset with a cause, effect and weight collumn.
    This dataset is createad from the cause_effect_pair.csv.

    Args:
        data (pd.DataFrame): The cause_effect_pair.csv dataset (needs to be loaded with load_cause_effect_pairs())
        fn (Callable[[CauseEffectPair, str], str]): The function how to create the nodes of the graph
        weight_threshold (int, optional): Filter the edges with a weight < the threshold.
        Defaults to 1 => keep all of the edges.

    Returns:
        pd.DataFrame: A dataframe cause, effect and weight collumn. Where cause, effect are the nodes of an edge and
        weight represents how often times the edges is mentioned.
    """
    graph = pd.DataFrame()
    pairs = data["pairs"].explode()
    graph["cause"] = pairs.apply(fn, select="cause")
    graph["effect"] = pairs.apply(fn, select="effect")
    graph = (
        graph.groupby(["cause", "effect"])
        .size()
        .sort_values(ascending=False)
        .reset_index(name="weight")
    )
    graph["reversed_weight"] = graph.apply(reversed_weight, graph=graph, axis=1)

    # Removes all edges that has a empty node during transformation
    emtpy_node_rule = (graph["cause"] != "") & (graph["effect"] != "")
    weight_threshold_rule = graph["weight"] >= weight_threshold
    graph = graph[emtpy_node_rule & weight_threshold_rule].reset_index(drop=True)

    # count the number of tokens per node
    nlp = English()
    tokenizer = Tokenizer(nlp.vocab)
    tokens_in_node = []
    seen = []
    for k,v in graph.iterrows():
        if v['cause'] not in seen:
            tokens_in_node.append(tokenizer(v['cause']))
            seen.append(v['cause'])
        if v['effect'] not in seen:
            tokens_in_node.append(tokenizer(v['effect']))
            seen.append(v['effect'])
    tokens_per_node = [len(i) for i in tokens_in_node]
    average_tokens_per_node = pd.Series(tokens_per_node).mean()

    return graph


def baseline_graph(pair: CauseEffectPair, select: str) -> str:
    if select == "cause":
        return " ".join((token.text for token in pair.cause))
    return " ".join((token.text for token in pair.effect))


def lemma_graph(pair: CauseEffectPair, select: str) -> str:
    if select == "cause":
        return " ".join((token.lemma.lower() for token in pair.cause))
    return " ".join((token.lemma.lower() for token in pair.effect))


def pos_graph(pair: CauseEffectPair, select: str) -> str:
    if select == "cause":
        return " ".join(
            (
                token.lemma.lower()
                for token in pair.cause
                if token.pos in PHRASE_REQUIRED_POS
            )
        )
    return " ".join(
        (
            token.lemma.lower()
            for token in pair.effect
            if token.pos in PHRASE_REQUIRED_POS
        )
    )


def root_graph(pair: CauseEffectPair, select: str) -> str:
    if select == "cause":
        return pair.cause_root.lemma.lower() if pair.cause_root else ""
    return pair.effect_root.lemma.lower() if pair.effect_root else ""


def build_graphs():
    pairs = load_cause_effect_pairs()

    graphs = [
        ("baseline", baseline_graph),
        ("lemma", lemma_graph),
        ("pos", pos_graph),
        ("root", root_graph),
    ]

    for name, fn in graphs:
        graph = create_graph(data=pairs, fn=fn, weight_threshold=1)
        save_data(graph, f"pipeline/05_build/{name}.csv")


if __name__ == "__main__":
    build_graphs()
