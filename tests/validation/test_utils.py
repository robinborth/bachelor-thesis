from src.utils import CauseEffectPairMeta, CauseEffectPairs, LabeledToken


def test_json_token_to_labeled_token():
    token_json = {
        "i": 0,
        "text": "Increasing",
        "pos": "VERB",
        "dep": "amod",
        "lemma": "increase",
        "labels": ["C1"],
    }
    assert LabeledToken(**token_json)


def test_cause_effect_pairs_len():
    json_tokens = [
        {
            "i": 0,
            "text": "Increasing",
            "pos": "VERB",
            "dep": "amod",
            "lemma": "increase",
            "labels": ["C1"],
        },
        {
            "i": 1,
            "text": "global",
            "pos": "ADJ",
            "dep": "amod",
            "lemma": "global",
            "labels": ["C1"],
        },
        {
            "i": 2,
            "text": "inequality",
            "pos": "NOUN",
            "dep": "nsubj",
            "lemma": "inequality",
            "labels": ["C1"],
        },
        {
            "i": 3,
            "text": "causes",
            "pos": "VERB",
            "dep": "ROOT",
            "lemma": "cause",
            "labels": ["I1"],
        },
        {
            "i": 4,
            "text": "migration",
            "pos": "NOUN",
            "dep": "dobj",
            "lemma": "migration",
            "labels": ["E1"],
        },
        {
            "i": 5,
            "text": ".",
            "pos": "PUNCT",
            "dep": "punct",
            "lemma": ".",
            "labels": ["O"],
        },
    ]
    tokens = [LabeledToken(**token) for token in json_tokens]
    pairs = CauseEffectPairs(tokens=tokens)
    assert len(pairs) == 1


def test_cause_effect_pairs_iter():
    json_tokens = [
        {
            "i": 0,
            "text": "Increasing",
            "pos": "VERB",
            "dep": "amod",
            "lemma": "increase",
            "labels": ["C1"],
        },
        {
            "i": 1,
            "text": "global",
            "pos": "ADJ",
            "dep": "amod",
            "lemma": "global",
            "labels": ["C1"],
        },
        {
            "i": 2,
            "text": "inequality",
            "pos": "NOUN",
            "dep": "nsubj",
            "lemma": "inequality",
            "labels": ["C1"],
        },
        {
            "i": 3,
            "text": "causes",
            "pos": "VERB",
            "dep": "ROOT",
            "lemma": "cause",
            "labels": ["I1"],
        },
        {
            "i": 4,
            "text": "migration",
            "pos": "NOUN",
            "dep": "dobj",
            "lemma": "migration",
            "labels": ["E1"],
        },
        {
            "i": 5,
            "text": ".",
            "pos": "PUNCT",
            "dep": "punct",
            "lemma": ".",
            "labels": ["O"],
        },
    ]

    tokens = [LabeledToken(**token) for token in json_tokens]
    meta = [CauseEffectPairMeta(trigger="simple_cause")]
    pairs = CauseEffectPairs(tokens=tokens, meta=meta)
    for index, pair in enumerate(pairs):
        assert pair.pair_id == 1

        true_cause_index = [0, 1, 2]
        for token, true_token in zip(pair.cause, true_cause_index):
            token.i == true_token

        true_effect_index = [4]
        for token, true_token in zip(pair.effect, true_effect_index):
            token.i == true_token

        assert pair.meta.trigger == "simple_cause"
