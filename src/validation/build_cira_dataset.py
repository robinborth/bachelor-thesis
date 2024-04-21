import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import spacy
from spacy.language import Language
from spacy.tokens import Token

from src.utils import Label, LabeledToken, load_data, save_data


@dataclass
class CiraLabel:
    name: str
    begin: int
    end: int


def labels2cirapairs(raw_lables: str) -> List[Tuple[List[CiraLabel], List[CiraLabel]]]:
    labels = [CiraLabel(**label) for label in json.loads(raw_lables)]
    pairs = defaultdict(list)
    for label in labels:
        pairs[label.name].append(label)

    cira_pairs: List[Tuple[List[CiraLabel], List[CiraLabel]]] = []
    if pairs["Cause1"] and pairs["Effect1"]:
        cira_pairs.append((pairs["Cause1"], pairs["Effect1"]))
    if pairs["Cause2"] and pairs["Effect2"]:
        cira_pairs.append((pairs["Cause2"], pairs["Effect2"]))
    if pairs["Cause3"] and pairs["Effect3"]:
        cira_pairs.append((pairs["Cause3"], pairs["Effect3"]))
    return cira_pairs


def is_token_in_phrase(token: Token, phrase: List[CiraLabel]) -> bool:
    begin = token.idx
    end = begin + len(token)
    for label in phrase:
        if label.begin <= begin <= label.end or label.begin <= end <= label.end:
            return True
    return False


def create_tokens(
    sentence: str,
    raw_labels: str,
    nlp: Language,
) -> str:
    doc = nlp(sentence)
    cira_pairs = labels2cirapairs(raw_labels)
    tokens: List[Dict] = []
    for token in doc:
        labels: List[str] = []

        for index, (causes, effects) in enumerate(cira_pairs):
            pair_id = index + 1
            if is_token_in_phrase(token, causes):
                labels.append(f"{Label.Cause.value}{pair_id}")
            if is_token_in_phrase(token, effects):
                labels.append(f"{Label.Effect.value}{pair_id}")

        if not labels:
            labels.append(Label.Other.value)

        labeled_token = LabeledToken(
            i=token.i,
            text=token.text,
            pos=token.pos_,
            dep=token.dep_,
            lemma=token.lemma_,
            labels=labels,
        )
        tokens.append(asdict(labeled_token))
    return json.dumps(tokens)


def build_cira_dataset() -> None:
    nlp = spacy.load("en_core_web_lg")
    cira_raw = load_data("validation/raw_cira/validation_cira_raw.csv")
    labels2tokens = lambda row: create_tokens(row["sentence"], row["labels"], nlp)
    cira_raw["tokens"] = cira_raw.apply(labels2tokens, axis=1)
    cira_raw.drop(columns=["labels"], inplace=True)
    save_data(cira_raw, "validation/validation_cira.csv")


if __name__ == "__main__":
    build_cira_dataset()
