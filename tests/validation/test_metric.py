import pandas as pd
import spacy

from src.pipeline.extract import convert2tokens
from src.utils import CauseEffectPairs
from src.validation.metric import (
    create_metric,
    harmonic_mean,
    iou_node,
    iou_pair,
    iou_pairs,
)


def test_harmonic_mean():
    assert harmonic_mean(1, 1) == 1
    assert harmonic_mean(2, 1) == harmonic_mean(1, 2)
    assert 0.685714 < harmonic_mean(0.6, 0.8) < 0.685715
    assert harmonic_mean(0, 0) == 0
    assert harmonic_mean(0, 1) == 0


def test_iou_node():
    source = [0, 1, 2, 5, 6, 8, 9]
    target = [0, 2, 3, 4, 5, 7, 9]
    assert iou_node(source, target) == iou_node(target, source)
    assert iou_node(source, target) == 0.4
    assert iou_node(source, []) == 0


def test_iou_pairs():
    nlp = spacy.load("en_core_web_lg")
    sentence = "A big and expensive crash was caused by an empty and defect battery."
    doc = nlp(sentence)

    source_pairs_text = [
        [["empty", "battery"], ["big", "crash"]],
        [["empty", "battery"], ["expensive", "crash"]],
        [["defect", "battery"], ["big", "crash"]],
        [["defect", "battery"], ["expensive", "crash"]],
    ]
    source_pairs = [
        [[doc[9], doc[12]], [doc[1], doc[4]]],
        [[doc[9], doc[12]], [doc[3], doc[4]]],
        [[doc[11], doc[12]], [doc[1], doc[4]]],
        [[doc[11], doc[12]], [doc[3], doc[4]]],
    ]
    for (cause, effect), (cause_text, effect_text) in zip(
        source_pairs, source_pairs_text
    ):
        for token, text in zip(cause, cause_text):
            assert token.text == text
        for token, text in zip(effect, effect_text):
            assert token.text == text
    source_pairs = CauseEffectPairs(tokens=convert2tokens(doc, source_pairs))

    target_pairs_text = [
        [["battery"], ["A", "big", "crash"]],
        [["empty", "battery"], ["expensive", "crash"]],
        [["big", "crash"], ["battery"]],
    ]
    target_pairs = [
        [[doc[12]], [doc[0], doc[1], doc[4]]],
        [[doc[9], doc[12]], [doc[3], doc[4]]],
        [[doc[1], doc[4]], [doc[12]]],
    ]

    for (cause, effect), (cause_text, effect_text) in zip(
        target_pairs, target_pairs_text
    ):
        for token, text in zip(cause, cause_text):
            assert token.text == text
        for token, text in zip(effect, effect_text):
            assert token.text == text
    target_pairs = CauseEffectPairs(tokens=convert2tokens(doc, target_pairs))

    pair_iou = iou_pair(next(iter(source_pairs)), next(iter(target_pairs)))
    assert pair_iou == harmonic_mean(1 / 2, 2 / 3)

    source_iou = iou_pairs(source_pairs, target_pairs)
    assert len(source_iou) == len(source_pairs)
    assert source_iou[0] == harmonic_mean(1 / 2, 2 / 3)
    assert source_iou[1] == harmonic_mean(2 / 2, 2 / 2)
    assert source_iou[2] == harmonic_mean(1 / 2, 2 / 3)
    assert source_iou[3] == harmonic_mean(1 / 3, 2 / 2)

    target_iou = iou_pairs(target_pairs, source_pairs)
    assert len(target_iou) == len(target_pairs)
    assert target_iou[0] == harmonic_mean(1 / 2, 2 / 3)
    assert target_iou[1] == harmonic_mean(2 / 2, 2 / 2)
    assert target_iou[2] == harmonic_mean(0 / 2, 0 / 1)


def test_create_metric():
    nlp = spacy.load("en_core_web_lg")
    sentence = "A big and expensive crash was caused by an empty and defect battery."
    doc = nlp(sentence)

    source_pairs_text = [
        [["empty", "battery"], ["big", "crash"]],
        [["empty", "battery"], ["expensive", "crash"]],
        [["defect", "battery"], ["big", "crash"]],
        [["defect", "battery"], ["expensive", "crash"]],
    ]
    source_pairs = [
        [[doc[9], doc[12]], [doc[1], doc[4]]],
        [[doc[9], doc[12]], [doc[3], doc[4]]],
        [[doc[11], doc[12]], [doc[1], doc[4]]],
        [[doc[11], doc[12]], [doc[3], doc[4]]],
    ]
    for (cause, effect), (cause_text, effect_text) in zip(
        source_pairs, source_pairs_text
    ):
        for token, text in zip(cause, cause_text):
            assert token.text == text
        for token, text in zip(effect, effect_text):
            assert token.text == text
    source_pairs = CauseEffectPairs(tokens=convert2tokens(doc, source_pairs))

    target_pairs_text = [
        [["battery"], ["A", "big", "crash"]],
        [["empty", "battery"], ["expensive", "crash"]],
        [["big", "crash"], ["battery"]],
    ]
    target_pairs = [
        [[doc[12]], [doc[0], doc[1], doc[4]]],
        [[doc[9], doc[12]], [doc[3], doc[4]]],
        [[doc[1], doc[4]], [doc[12]]],
    ]

    for (cause, effect), (cause_text, effect_text) in zip(
        target_pairs, target_pairs_text
    ):
        for token, text in zip(cause, cause_text):
            assert token.text == text
        for token, text in zip(effect, effect_text):
            assert token.text == text
    target_pairs = CauseEffectPairs(tokens=convert2tokens(doc, target_pairs))

    ground_truths = pd.Series([source_pairs])
    predictions = pd.Series([target_pairs])
    assert len(ground_truths) == len(predictions) == 1
    thresholds = [0.45, 0.55, 0.65]
    metrics = create_metric(ground_truths, predictions, thresholds)
    assert len(metrics) == len(thresholds)
    # true_positive + false_negative needs to be len(source_pairs)
    for _, metric in metrics.iterrows():
        if metric["threshold"] == 0.45:
            assert metric["true_negative"] == 0
            assert metric["true_positive"] == 4
            assert metric["false_negative"] == 0
            assert metric["false_positive"] == 1
        if metric["threshold"] == 0.55:
            assert metric["true_negative"] == 0
            assert metric["true_positive"] == 3
            assert metric["false_negative"] == 1
            assert metric["false_positive"] == 1
        if metric["threshold"] == 0.65:
            assert metric["true_negative"] == 0
            assert metric["true_positive"] == 1
            assert metric["false_negative"] == 3
            assert metric["false_positive"] == 2

    ground_truths, predictions = predictions, ground_truths
    assert len(ground_truths) == len(predictions) == 1
    thresholds = [0.45, 0.55, 0.65]
    metrics = create_metric(ground_truths, predictions, thresholds)
    assert len(metrics) == len(thresholds)
    # true_positive + false_negative needs to be len(source_pairs)
    for _, metric in metrics.iterrows():
        if metric["threshold"] == 0.45:
            assert metric["true_negative"] == 0
            assert metric["true_positive"] == 2
            assert metric["false_negative"] == 1
            assert metric["false_positive"] == 0
        if metric["threshold"] == 0.55:
            assert metric["true_negative"] == 0
            assert metric["true_positive"] == 2
            assert metric["false_negative"] == 1
            assert metric["false_positive"] == 1
        if metric["threshold"] == 0.65:
            assert metric["true_negative"] == 0
            assert metric["true_positive"] == 1
            assert metric["false_negative"] == 2
            assert metric["false_positive"] == 3
