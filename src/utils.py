import json
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Generator, Iterable, List, Optional

import networkx as nx
import pandas as pd
from matplotlib.figure import Figure
from networkx import Graph


class Annotator(Enum):
    EhsanZibaei = "ez"
    RobinBorth = "rb"
    Cira = "cira"


class Label(Enum):
    Cause = "C"
    Effect = "E"
    Other = "O"


@dataclass
class LabeledToken:
    i: int
    text: str
    pos: str
    dep: str
    lemma: str
    labels: List[str]

    @property
    def highest_pair_id(self) -> int:
        highest_pair_count = 0
        for label in self.labels:
            match = re.match(r"(?:C|O|E)(\d+)", label)
            if match:
                pair_count = int(match.group(1))
                highest_pair_count = max(highest_pair_count, pair_count)
        return highest_pair_count


@dataclass
class CauseEffectPairMeta:
    trigger: Optional[str] = None
    cause_root_index: Optional[int] = None
    effect_root_index: Optional[int] = None


@dataclass
class CauseEffectPair:
    pair_id: int
    cause: List[LabeledToken]
    effect: List[LabeledToken]
    meta: CauseEffectPairMeta

    @property
    def cause_as_index(self) -> List[int]:
        return [token.i for token in self.cause]

    @property
    def effect_as_index(self) -> List[int]:
        return [token.i for token in self.effect]

    @property
    def cause_root(self) -> Optional[LabeledToken]:
        for token in self.cause:
            if self.meta and token.i == self.meta.cause_root_index:
                return token
        return None

    @property
    def effect_root(self) -> Optional[LabeledToken]:
        for token in self.effect:
            if self.meta and token.i == self.meta.effect_root_index:
                return token
        return None


@dataclass
class CauseEffectPairs:
    tokens: List[LabeledToken]
    meta: List[CauseEffectPairMeta] = field(default_factory=list)

    def __bool__(self):
        return bool(len(self))

    def __len__(self):
        return max((token.highest_pair_id for token in self.tokens))

    def __iter__(self):
        self.id = 1
        return self

    def __next__(self) -> CauseEffectPair:
        if self.id > len(self):
            raise StopIteration

        meta = CauseEffectPairMeta()
        if self.meta:
            meta = self.meta[self.id - 1]

        pair = CauseEffectPair(
            pair_id=self.id,
            cause=[
                token
                for token in self.tokens
                if f"{Label.Cause.value}{self.id}" in token.labels
            ],
            effect=[
                token
                for token in self.tokens
                if f"{Label.Effect.value}{self.id}" in token.labels
            ],
            meta=meta,
        )

        self.id += 1

        return pair


BASE_PATH = Path(__file__).parent.parent
BASE_ANNOTATOR = Annotator.RobinBorth


def save_data(df: pd.DataFrame, path: str) -> None:
    df.to_csv(BASE_PATH / "data" / path, index=False)


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(BASE_PATH / "data" / path)


def save_report(text: str, path: str) -> None:
    with open(BASE_PATH / "results/reports" / path, "w") as f:
        f.write(text)


def exists_data(path: str) -> bool:
    return os.path.exists(BASE_PATH / "data" / path)


def load_source() -> pd.DataFrame:
    return load_data("pipeline/02_preprocess/source.csv")


def load_prefiltered() -> pd.DataFrame:
    prefiltered = load_data("pipeline/03_filter/prefiltered.csv")
    prefiltered["triggers"] = prefiltered["triggers"].apply(json.loads)
    return prefiltered


def load_cause_effect_pairs() -> pd.DataFrame:
    pairs = load_data("pipeline/04_extract/cause_effect_pairs.csv")

    def load_tokens_with_meta_as_pairs(data: Any) -> CauseEffectPairs:
        tokens = [LabeledToken(**token) for token in json.loads(data["tokens"])]
        meta = [CauseEffectPairMeta(**meta) for meta in json.loads(data["meta"])]
        return CauseEffectPairs(tokens=tokens, meta=meta)

    pairs["pairs"] = pairs.apply(load_tokens_with_meta_as_pairs, axis=1)
    return pairs


def load_graph(graph: str) -> Graph:
    data = load_data(f"pipeline/05_build/{graph}.csv")
    G = nx.Graph()
    for _, row in data.iterrows():
        (u, v), weight = (row["cause"], row["effect"]), row["weight"]
        if (u, v) in G.edges or (v, u) in G.edges:
            current_weight = G[u][v]["weight"]
            G.add_edge(u, v, weight=current_weight + weight)
        else:
            G.add_edge(u, v, weight=weight)
    return G

def load_digraph(graph: str) -> Graph:
    data = load_data(f"pipeline/05_build/{graph}.csv")
    G = nx.DiGraph()
    for _, row in data.iterrows():
        (u, v) = (row["cause"], row["effect"])
        G.add_edge(u, v)
    return G

def load_validation(annotator: Annotator) -> pd.DataFrame:
    validation = load_data(f"validation/validation_{annotator.value}.csv")

    def load_tokens_as_pairs(tokens: Any) -> CauseEffectPairs:
        return CauseEffectPairs([LabeledToken(**token) for token in json.loads(tokens)])

    validation["ground_truths"] = validation["tokens"].apply(load_tokens_as_pairs)
    return validation


def load_raw_validation(annotator: Annotator) -> pd.DataFrame:
    path = BASE_PATH / "data" / "validation" / f"raw_{annotator.value}"
    val_data = []
    for val_file in sorted(os.listdir(path)):
        if val_file.endswith(f"{annotator.value}.json"):
            val_slice_data = pd.read_json(path / val_file)
            val_data.append(pd.json_normalize(val_slice_data["items"]))
    return pd.concat(val_data).drop(columns=["index"]).reset_index(drop=True)


def save_plot(figure: Figure, path: str) -> None:
    figure.savefig(str(BASE_PATH / "results/plots" / path))


def deep_flatten(container: Iterable[Any]) -> Generator[Any, None, None]:
    for item in container:
        if isinstance(item, (list, tuple)):
            yield from deep_flatten(item)
        else:
            yield item
