import functools
import json
from dataclasses import asdict
from itertools import product
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple

import pandas as pd
import spacy
from spacy.language import Language
from spacy.matcher import DependencyMatcher
from spacy.tokens import Doc, Token
from tqdm import tqdm

from src.pipeline.filter import CAUSALITY_TRIGGERS, prefilter
from src.utils import (
    CauseEffectPairMeta,
    CauseEffectPairs,
    Label,
    LabeledToken,
    load_data,
    load_prefiltered,
    save_data,
)

BASE_PATTERNS = {
    "simple": [
        {
            "LEFT_ID": "verb",
            "REL_OP": ">",
            "RIGHT_ID": "cause",
            "RIGHT_ATTRS": {"DEP": "nsubj"},
        },
        {
            "LEFT_ID": "verb",
            "REL_OP": ">",
            "RIGHT_ID": "effect",
            "RIGHT_ATTRS": {"DEP": "dobj"},
        },
    ],
    "phrasal": [
        {
            "LEFT_ID": "verb",
            "REL_OP": ">",
            "RIGHT_ID": "prep",
            "RIGHT_ATTRS": {"LEMMA": {"IN": ["in", "of", "to"]}},
        },
        {
            "LEFT_ID": "verb",
            "REL_OP": ">",
            "RIGHT_ID": "cause",
            "RIGHT_ATTRS": {"DEP": "nsubj"},
        },
        {
            "LEFT_ID": "prep",
            "REL_OP": ">",
            "RIGHT_ID": "effect",
            "RIGHT_ATTRS": {"DEP": "pobj"},
        },
    ],
    "passive": [
        {
            "LEFT_ID": "verb",
            "REL_OP": ">",
            "RIGHT_ID": "prep",
            "RIGHT_ATTRS": {
                "DEP": "agent",
                "LEMMA": {"IN": ["by"]},
            },
        },
        {
            "LEFT_ID": "prep",
            "REL_OP": ">",
            "RIGHT_ID": "cause",
            "RIGHT_ATTRS": {"DEP": "pobj"},
        },
        {
            "LEFT_ID": "verb",
            "REL_OP": ">",
            "RIGHT_ID": "effect",
            "RIGHT_ATTRS": {"DEP": "nsubjpass"},
        },
    ],
}


class CauseEffectPairMatcher:
    def __init__(
        self,
        nlp: Language,
        triggers: List[str] = CAUSALITY_TRIGGERS,
        base_patterns: Dict[str, Any] = BASE_PATTERNS,
    ) -> None:
        self.nlp = nlp
        self.matcher = DependencyMatcher(nlp.vocab, validate=True)
        for key, item in base_patterns.items():
            for trigger in triggers:
                pattern_key = f"{key}_{trigger}"
                pattern_value = [
                    {
                        "RIGHT_ID": "verb",
                        "RIGHT_ATTRS": {"LEMMA": trigger},
                    },
                    *item,
                ]
                self.matcher.add(pattern_key, [pattern_value])

    def __call__(self, doc: Doc) -> Generator[Tuple[str, Token, Token], None, None]:
        for match_id, token_ids in self.matcher(doc):
            trigger = str(self.nlp.vocab.strings[match_id])
            pattern = self.matcher.get(trigger)[1][0]
            cause_root = doc[int(token_ids[len(pattern) - 2])]
            effect_root = doc[int(token_ids[len(pattern) - 1])]
            yield trigger, cause_root, effect_root


PHRASE_FOLLOW_DEPS = [
    "csubjpass",
    "nsubjpass",
    "nsubj",
    "csubj",
    "dobj",
    "pobj",
    "nn",
    "amod",
    "nmod",
    "advmod",
    "compound",
    "prep",
    "poss",
]


def follow_children(token: Token) -> Generator[Token, None, None]:
    for children in token.children:
        if children.dep_ in PHRASE_FOLLOW_DEPS:
            yield children


def conj_children(token: Token) -> Generator[Token, None, None]:
    for children in token.children:
        if children.dep_ == "conj":
            yield children


def deep_flatten(container: Iterable[Any]) -> Generator[Any, None, None]:
    for item in container:
        if isinstance(item, (list, tuple)):
            yield from deep_flatten(item)
        else:
            yield item


def is_leaf(token: Token) -> bool:
    return not token.children


def merge_phrases(p1: List[List[Token]], p2: List[List[Token]]) -> List[List[Token]]:
    return [list(deep_flatten(phrase)) for phrase in product(p1, p2)]


def combine_with_token(token: Token, phrases: List[List[Token]]) -> List[List[Token]]:
    return [sorted(phrase + [token], key=lambda t: t.i) for phrase in phrases]


def extract_phrases(token: Token) -> List[List[Token]]:
    if is_leaf(token):
        return [[token]]

    phrases: List[List[Token]] = []
    children = list(follow_children(token))
    if children:
        children_phrases = [extract_phrases(child) for child in children]
        merged_phrases = functools.reduce(merge_phrases, children_phrases)
        phrases.extend(combine_with_token(token, merged_phrases))
    else:
        phrases.extend([[token]])

    for child in conj_children(token):
        phrases.extend(extract_phrases(child))

    return phrases


PHRASE_STRIP_BY_TEXT = [
    "that",
    "these",
    "those",
    "such",
    "of",
    "in",
    "at",
    "by",
    "any",
    "some",
    "much",
    " ",
    "-",
]


PHRASE_FILTER_BY_POS = [
    "PRON",
]


PHRASE_FILTER_BY_DEP = [
    "cc",
]

PHRASE_FILTER_BY_TEXT = [
    "a",
    "an",
    "i",
    "the",
    "this",
]

PHRASE_REQUIRED_POS = [
    "NOUN",
    "PROPN",
    "VERB",
]


def strip_phrase(phrase: List[Token]) -> List[Token]:
    while phrase and phrase[0].lemma_.lower() in PHRASE_STRIP_BY_TEXT:
        phrase.pop(0)
    while phrase and phrase[-1].lemma_.lower() in PHRASE_STRIP_BY_TEXT:
        phrase.pop(-1)
    return phrase


def keep_phrase_token(token: Token) -> bool:
    return not (
        token.dep_ in PHRASE_FILTER_BY_DEP
        or token.lemma_.lower() in PHRASE_FILTER_BY_TEXT
        or token.pos_ in PHRASE_FILTER_BY_POS
    )


def phrase_has_required_token(phrase: List[Token]) -> bool:
    return any((token.pos_ in PHRASE_REQUIRED_POS for token in phrase))


def clear_phrases(phrases: List[List[Token]]) -> List[List[Token]]:
    cleared_phrases: List[List[Token]] = []
    for phrase in phrases:
        cleared_phrase = [token for token in phrase if keep_phrase_token(token)]
        stripped_phrase = strip_phrase(cleared_phrase)
        if stripped_phrase and phrase_has_required_token(stripped_phrase):
            cleared_phrases.append(stripped_phrase)
    return cleared_phrases


def merge_phrases_to_pairs(
    p1: List[List[Token]], p2: List[List[Token]]
) -> List[Tuple[List[Token], List[Token]]]:
    return list(product(p1, p2))


def convert2tokens(
    doc: Doc,
    pairs: List[Tuple[List[Token], List[Token]]],
) -> List[LabeledToken]:
    tokens: List[LabeledToken] = []
    for token in doc:
        labels: List[str] = []
        for index, (cause, effect) in enumerate(pairs):
            pair_id = index + 1
            if token in cause:
                labels.append(f"{Label.Cause.value}{pair_id}")
            if token in effect:
                labels.append(f"{Label.Effect.value}{pair_id}")
        if not labels:
            labels = [Label.Other.value]

        labeled_token = LabeledToken(
            i=token.i,
            text=token.text,
            pos=token.pos_,
            dep=token.dep_,
            lemma=token.lemma_,
            labels=labels,
        )
        tokens.append(labeled_token)
    return tokens


def extract_cause_effect_pairs(
    doc: Doc,
    matcher: CauseEffectPairMatcher,
) -> CauseEffectPairs:
    """Returns cause and effect phrases as a list of tokens"""

    pairs: List[Tuple[List[Token], List[Token]]] = []
    metas: List[CauseEffectPairMeta] = []

    if not prefilter(doc):
        # 1) Find the root cause effect token from a sentence
        for trigger, cause_root, effect_root in matcher(doc):
            # 2) Extract the phrases that are there based on the root words
            cause_phrases = extract_phrases(cause_root)
            effect_phrases = extract_phrases(effect_root)

            # 3) Clear the cause effect pairs that messed something up while extracting the phrases
            cause_cleared_phrases = clear_phrases(cause_phrases)
            effect_cleared_phrases = clear_phrases(effect_phrases)

            # 4) Combine each cause phrase with each effect pair
            merged_pairs = merge_phrases_to_pairs(
                cause_cleared_phrases,
                effect_cleared_phrases,
            )

            pairs.extend(merged_pairs)

            meta = CauseEffectPairMeta(
                trigger=trigger,
                cause_root_index=int(cause_root.i),
                effect_root_index=int(effect_root.i),
            )
            metas.extend([meta] * len(merged_pairs))

    tokens = convert2tokens(doc=doc, pairs=pairs)
    return CauseEffectPairs(tokens=tokens, meta=metas)


def extract_tokens_with_meta(
    source: pd.DataFrame,
    n_process=4,
) -> Tuple[List[str], List[str], List[bool]]:
    nlp = spacy.load("en_core_web_lg")
    nlp_pipe = nlp.pipe(source["sentence"], batch_size=50, n_process=n_process)
    matcher = CauseEffectPairMatcher(nlp=nlp)

    tokens: List[str] = []
    metas: List[str] = []
    indexes: List[bool] = []

    for doc in tqdm(nlp_pipe, total=len(source)):
        pairs = extract_cause_effect_pairs(doc, matcher)
        tokens.append(json.dumps([asdict(token) for token in pairs.tokens]))
        metas.append(json.dumps([asdict(meta) for meta in pairs.meta]))
        indexes.append(bool(pairs))

    return tokens, metas, indexes


def main():
    source = load_prefiltered()
    source["tokens"], source["meta"], indexes = extract_tokens_with_meta(source)
    source = source.loc[indexes, ["id", "sentence", "source", "tokens", "meta"]]
    save_data(source, "pipeline/04_extract/cause_effect_pairs.csv")


if __name__ == "__main__":
    main()
