import json
from typing import List

import pandas as pd
import spacy
from spacy.matcher import Matcher
from spacy.tokens import Doc
from tqdm import tqdm

from src.utils import load_data, load_source, save_data

CAUSALITY_TRIGGERS = [
    "cause",
    "generate",
    "trigger",
    "induce",
    "produce",
    "effect",
    "provoke",
    "arouse",
    "elicit",
    "lead",
    "derive",
    "associate",
    "relate",
    "link",
    "stem",
    "iginate",
    "result",
    "entail",
    "commence",
    "spark",
    "evoke",
    "implicate",
    "activate",
    "actuate",
    "kindle",
    "stimulate",
    "unleash",
    "effectuate",
]


INTERROGATIVE_PRONOUNS = [
    "what",
    "why",
    "who",
    "whom",
    "whose",
    "which",
    "how",
]


def prefilter(doc: Doc) -> bool:
    def has_negation() -> bool:
        return any((token.dep_ == "neg" for token in doc))

    def has_question_mark() -> bool:
        return any((token.lemma_ == "?" for token in doc))

    def has_interrogative_pronoun() -> bool:
        return any((token.lemma_ in INTERROGATIVE_PRONOUNS for token in doc))

    return any([has_negation(), has_question_mark(), has_interrogative_pronoun()])


def find_triggers(source: pd.DataFrame, n_process=4) -> List[List[str]]:
    nlp = spacy.load("en_core_web_sm", exclude=["ner", "parser"])
    nlp_pipe = nlp.pipe(source["sentence"], batch_size=50, n_process=n_process)
    matcher = Matcher(nlp.vocab, validate=True)
    for key in CAUSALITY_TRIGGERS:
        matcher.add(key, [[{"LEMMA": key}]])

    source_triggers: List[List[str]] = []
    for doc in tqdm(nlp_pipe, total=len(source)):
        if prefilter(doc):
            source_triggers.append([])
        else:
            matches = matcher(doc)
            triggers = [str(nlp.vocab.strings[match[0]]) for match in matches]  # type: ignore
            source_triggers.append(triggers)

    return source_triggers


def main():
    source = load_source()
    source["triggers"] = find_triggers(source)
    source = source[source["triggers"].apply(len) > 0]
    source["triggers"] = source["triggers"].apply(json.dumps)
    save_data(source, "pipeline/03_filter/prefiltered.csv")


if __name__ == "__main__":
    main()
