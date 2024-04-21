import json
import os
import random
import re
from typing import Any, Dict, List

import spacy

from src.utils import (
    BASE_ANNOTATOR,
    BASE_PATH,
    Annotator,
    load_data,
    load_raw_validation,
)
from src.validation.build_datasets import build_datasets


# The different sources where to get the datafrom
# Needs to define an output format that is used as input for the create validation dataset
# The functions need to transform the data into the correct format
def load_causal_sentences() -> List[str]:
    data = load_data("pipeline/03_filter/prefiltered.csv")
    rule = data["triggers"].apply(lambda identifiers: "cause" in identifiers)
    return data[rule]["sentence"].to_list()


def load_cause_effect_pair_sentences() -> List[str]:
    raise NotImplementedError()


# The rules that are applied for each source e.g. word counts or is the sentence in the validation set.
# Takes the data from the source and returns the dataset thats need to be sampled for
def apply_rules(source: List[str]):
    labeled_data = load_raw_validation(BASE_ANNOTATOR)
    labeled_sentences = labeled_data["sentence"].to_list()

    def is_in_word_range_rule(sentence: str) -> bool:
        word_count = len(sentence.split())
        return (word_count > 8) and (word_count < 30)

    def is_not_labeled_rule(sentence: str) -> bool:
        return sentence not in labeled_sentences

    return [
        sentence
        for sentence in source
        if all([is_in_word_range_rule(sentence), is_not_labeled_rule(sentence)])
    ]


# Samples the dataset
def next_validation_item_id() -> int:
    labeled_data = load_raw_validation(BASE_ANNOTATOR)
    return int(labeled_data["id"].max() + 1)


def next_validation_file_id() -> str:
    path = BASE_PATH / "data" / "validation" / f"raw_{BASE_ANNOTATOR.value}"

    def file2id(val_file: str) -> int:
        match = re.match(r"(?:TODO_)?(\d+)*.", val_file)
        return int(match.group(1)) if match else -1

    max_id = max([file2id(val_file) for val_file in os.listdir(path)])
    return str(max_id + 1).zfill(2)


# The validation scema that is used for the webapp:
def sample2labels(sample: List[str]) -> Dict[str, Any]:
    nlp = spacy.load("en_core_web_lg")
    base_id = next_validation_item_id()
    docs = [nlp(sentence) for sentence in sample]
    return {
        "itemIndex": 0,
        "pairId": 1,
        "editMode": "C",
        "mouseState": "up",
        "searchState": "all",
        "tooltip": False,
        "items": [
            {
                "id": base_id + index,
                "index": index,
                "sentence": doc.text,
                "foreignDomain": False,
                "ignore": False,
                "labelState": "pending",
                "frequencyState": "none",
                "tokens": [
                    {
                        "i": token.i,
                        "text": token.text,
                        "pos": token.pos_,
                        "dep": token.dep_,
                        "lemma": token.lemma_,
                        "labels": ["O"],
                    }
                    for token in doc
                ],
            }
            for index, doc in enumerate(docs)
        ],
    }


def save_new_labels(labels: Dict[str, Any], annotator: Annotator, file_id: str) -> None:
    path = BASE_PATH / "data" / "validation" / f"raw_{annotator.value}"

    file_name = (
        f"TODO_{file_id}_ardupilot_{len(labels['items'])}_{annotator.value}.json"
    )
    with open(path / file_name, "w") as f:
        json.dump(labels, f, indent=2)


def create_unlabeled_data(source: List[str], sample_size: int):
    filtered_source = apply_rules(source)
    sample = random.sample(filtered_source, sample_size)
    labels = sample2labels(sample)
    next_file_id = next_validation_file_id()
    # Save for both Annotators
    save_new_labels(labels, Annotator.RobinBorth, next_file_id)
    save_new_labels(labels, Annotator.EhsanZibaei, next_file_id)


if __name__ == "__main__":
    build_datasets()
    create_unlabeled_data(
        source=load_causal_sentences(),
        sample_size=200,
    )
