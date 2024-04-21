import spacy

from src.pipeline.extract import (
    CauseEffectPairMatcher,
    extract_cause_effect_pairs,
    extract_phrases,
)


def test_cause_effect_matcher():
    nlp = spacy.load("en_core_web_lg")
    matcher = CauseEffectPairMatcher(nlp=nlp)
    example_sent = "The empty battery causes a crash!"
    doc = nlp(example_sent)

    # There is only one cause effect pair in the sentences
    matches = list(matcher(doc))
    assert len(matches) == 1

    trigger, cause_root, effect_root = matches[0]
    assert trigger == "simple_cause"
    assert cause_root.text == "battery"
    assert effect_root.text == "crash"


def test_extract_phrase_simple_case():
    nlp = spacy.load("en_core_web_lg")
    sentence = "An empty and defect battery causes a crash."
    doc = nlp(sentence)

    cause_root_text = "battery"
    cause_root = doc[4]
    assert cause_root.text == cause_root_text

    true_phrases_text = [["empty", "battery"], ["defect", "battery"]]
    true_phrases = [[doc[1], doc[4]], [doc[3], doc[4]]]
    for phrase, phrase_text in zip(true_phrases, true_phrases_text):
        for token, text in zip(phrase, phrase_text):
            assert token.text == text

    phrases = extract_phrases(cause_root)
    assert len(phrases) == len(true_phrases)
    for phrase, true_phrase in zip(phrases, true_phrases):
        assert len(phrase) == len(true_phrase)
        for token, true_token in zip(phrase, true_phrase):
            assert token.i == true_token.i


def test_extract_phrase_complex_case():
    nlp = spacy.load("en_core_web_lg")
    sentence = """An empty and defect battery, GPS fault or defect motor of UAV causes a crash."""
    doc = nlp(sentence)
    cause_root = doc[4]

    cause_root_text = "battery"
    assert cause_root.text == cause_root_text

    true_phrases_text = [
        ["empty", "battery"],
        ["defect", "battery"],
        ["GPS", "fault"],
        ["defect", "motor", "of", "UAV"],
    ]
    true_phrases = [
        [doc[1], doc[4]],
        [doc[3], doc[4]],
        [doc[6], doc[7]],
        [doc[9], doc[10], doc[11], doc[12]],
    ]
    for phrase, phrase_text in zip(true_phrases, true_phrases_text):
        for token, text in zip(phrase, phrase_text):
            assert token.text == text

    phrases = extract_phrases(cause_root)
    assert len(phrases) == len(true_phrases)
    for phrase, true_phrase in zip(phrases, true_phrases):
        assert len(phrase) == len(true_phrase)
        for token, true_token in zip(phrase, true_phrase):
            assert token.i == true_token.i


def test_extract_cause_effect_pairs_simple_cause():
    nlp = spacy.load("en_core_web_lg")
    sentence = "An empty and defect battery causes a crash."
    matcher = CauseEffectPairMatcher(nlp=nlp)
    doc = nlp(sentence)

    true_triggers = ["simple_cause", "simple_cause"]
    true_pairs_text = [
        [["empty", "battery"], ["crash"]],
        [["defect", "battery"], ["crash"]],
    ]
    true_pairs = [
        [[doc[1], doc[4]], [doc[7]]],
        [[doc[3], doc[4]], [doc[7]]],
    ]
    for (cause, effect), (cause_text, effect_text) in zip(true_pairs, true_pairs_text):
        for token, text in zip(cause, cause_text):
            assert token.text == text
        for token, text in zip(effect, effect_text):
            assert token.text == text

    pairs = extract_cause_effect_pairs(doc, matcher)
    assert len(pairs.meta) == len(true_triggers)
    for pair, true_trigger in zip(pairs, true_triggers):
        assert pair.meta.trigger == true_trigger

    assert len(pairs) == len(true_pairs)
    for pair, (true_cause, true_effect) in zip(pairs, true_pairs):
        for token, true_token in zip(pair.cause, true_cause):
            assert token.i == true_token.i
            assert token.text == true_token.text
        for token, true_token in zip(pair.effect, true_effect):
            assert token.i == true_token.i
            assert token.text == true_token.text


def test_extract_cause_effect_pairs_passive_cause_complex():
    nlp = spacy.load("en_core_web_lg")
    sentence = "A big and expensive crash was caused by an empty and defect battery."
    matcher = CauseEffectPairMatcher(nlp=nlp)
    doc = nlp(sentence)

    true_triggers = [
        "passive_cause",
        "passive_cause",
        "passive_cause",
        "passive_cause",
    ]
    true_pairs_text = [
        [["empty", "battery"], ["big", "crash"]],
        [["empty", "battery"], ["expensive", "crash"]],
        [["defect", "battery"], ["big", "crash"]],
        [["defect", "battery"], ["expensive", "crash"]],
    ]
    true_pairs = [
        [[doc[9], doc[12]], [doc[1], doc[4]]],
        [[doc[9], doc[12]], [doc[3], doc[4]]],
        [[doc[11], doc[12]], [doc[1], doc[4]]],
        [[doc[11], doc[12]], [doc[3], doc[4]]],
    ]
    for (cause, effect), (cause_text, effect_text) in zip(true_pairs, true_pairs_text):
        for token, text in zip(cause, cause_text):
            assert token.text == text
        for token, text in zip(effect, effect_text):
            assert token.text == text

    pairs = extract_cause_effect_pairs(doc, matcher)
    assert len(pairs.meta) == len(true_triggers)
    for pair, true_trigger in zip(pairs, true_triggers):
        assert pair.meta.trigger == true_trigger

    assert len(pairs) == len(true_pairs)
    for pair, (true_cause, true_effect) in zip(pairs, true_pairs):
        for token, true_token in zip(pair.cause, true_cause):
            assert token.i == true_token.i
            assert token.text == true_token.text
        for token, true_token in zip(pair.effect, true_effect):
            assert token.i == true_token.i
            assert token.text == true_token.text
