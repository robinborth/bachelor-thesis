from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from src.utils import CauseEffectPair, CauseEffectPairs


def harmonic_mean(x: float, y: float) -> float:
    return (2 * x * y) / (x + y) if (x + y) else 0


def iou_node(source: List[int], target: List[int]) -> float:
    return len(set(source) & set(target)) / len(set(source) | set(target))


def iou_pair(source: CauseEffectPair, target: CauseEffectPair) -> float:
    iou_cause = iou_node(source.cause_as_index, target.cause_as_index)
    iou_effect = iou_node(source.effect_as_index, target.effect_as_index)
    return harmonic_mean(iou_cause, iou_effect)


def iou_pairs(source: CauseEffectPairs, target: CauseEffectPairs) -> List[float]:
    iou_pairs: List[float] = []
    for source_pair in source:
        iou: float = 0
        for target_pair in target:
            iou = max(iou_pair(source_pair, target_pair), iou)
        iou_pairs.append(iou)
    return iou_pairs


def true_negative(row: Any) -> int:
    return int(not row["iou_ground_truths"] and not row["iou_predictions"])


def true_positive(row: Any, threshold: float) -> int:
    return sum((int(iou > threshold) for iou in row["iou_ground_truths"]))


def false_negative(row: Any, threshold: float) -> int:
    return sum((int(iou <= threshold) for iou in row["iou_ground_truths"]))


def false_positive(row: Any, threshold: float) -> int:
    return sum((int(iou <= threshold) for iou in row["iou_predictions"]))


def create_metric(
    ground_truths: pd.Series,
    predictions: pd.Series,
    thresholds: Optional[List[float]] = None,
) -> pd.DataFrame:

    if thresholds is None:
        thresholds = list(np.linspace(0.0, 0.99, num=100))

    data = pd.DataFrame()
    data["ground_truths"] = ground_truths
    data["predictions"] = predictions
    iou_ground = lambda item: iou_pairs(item["ground_truths"], item["predictions"])
    data["iou_ground_truths"] = data.apply(iou_ground, axis=1)
    iou_pred = lambda item: iou_pairs(item["predictions"], item["ground_truths"])
    data["iou_predictions"] = data.apply(iou_pred, axis=1)

    metrics: List[Dict[str, Union[int, float]]] = []
    for threshold in thresholds:
        metric: Dict[str, Union[int, float]] = {}
        tn = int(data.apply(true_negative, axis=1).sum())
        tp = int(data.apply(true_positive, threshold=threshold, axis=1).sum())
        fn = int(data.apply(false_negative, threshold=threshold, axis=1).sum())
        fp = int(data.apply(false_positive, threshold=threshold, axis=1).sum())
        metric["accuracy"] = (tp + tn) / (tp + tn + fp + fn)
        metric["precision"] = (tp) / (tp + fp)
        metric["recall"] = (tp) / (tp + fn)
        metric["f1"] = harmonic_mean(metric["precision"], metric["recall"])
        metric["threshold"] = threshold
        metric["true_negative"] = tn
        metric["true_positive"] = tp
        metric["false_negative"] = fn
        metric["false_positive"] = fp
        metrics.append(metric)

    return pd.DataFrame(metrics)


# def source_target_transformer(source, target):
#     if isinstance(source, str) and isinstance(target, str):
#         return source.split(), target.split()
#     return source, target


# def is_node(source, target, threshold):
#     return iou_node(source, target) > threshold


# def is_pair(source, target, threshold):
#     return iou_pair(source, target) > threshold
