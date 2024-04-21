from typing import Callable, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import spacy
from matplotlib.figure import Figure
from tqdm import tqdm

from src.pipeline.extract import (
    BASE_PATTERNS,
    CauseEffectPairMatcher,
    extract_cause_effect_pairs,
)
from src.utils import Annotator, load_validation, save_plot
from src.validation.metric import create_metric


def create_pattern_dataset(data: pd.DataFrame) -> pd.DataFrame:
    nlp = spacy.load("en_core_web_lg")
    data = data[data["is_ardu"]]

    pattern_combinations = [
        ["simple"],
        ["phrasal"],
        ["passive"],
        ["simple", "phrasal"],
        ["simple", "passive"],
        ["phrasal", "passive"],
        ["simple", "phrasal", "passive"],
    ]

    pattern_dataset = pd.DataFrame()
    for pattern in tqdm(pattern_combinations, total=len(pattern_combinations)):
        base_pattern = {key: BASE_PATTERNS[key] for key in pattern}
        matcher = CauseEffectPairMatcher(nlp=nlp, base_patterns=base_pattern)
        sent2pair = lambda sent: extract_cause_effect_pairs(nlp(sent), matcher)
        predictions = data["sentence"].apply(sent2pair)
        metric = create_metric(
            ground_truths=data["ground_truths"],
            predictions=predictions,
        )
        metric["hyperparameter"] = "_".join(pattern)
        pattern_dataset = pd.concat([pattern_dataset, metric])

    return pattern_dataset


def create_model_dataset(data: pd.DataFrame) -> pd.DataFrame:
    model_combinations = [
        "en_core_web_sm",
        "en_core_web_md",
        "en_core_web_lg",
    ]

    model_dataset = pd.DataFrame()
    for model in tqdm(model_combinations, total=len(model_combinations)):
        nlp = spacy.load(model)
        matcher = CauseEffectPairMatcher(nlp=nlp)
        sent2pair = lambda sent: extract_cause_effect_pairs(nlp(sent), matcher)
        predictions = data["sentence"].apply(sent2pair)
        metric = create_metric(
            ground_truths=data["ground_truths"],
            predictions=predictions,
        )
        metric["hyperparameter"] = model
        model_dataset = pd.concat([model_dataset, metric])

    return model_dataset


def create_ardu_nato_dataset(data: pd.DataFrame) -> pd.DataFrame:
    nlp = spacy.load("en_core_web_lg")

    ardu_nato_combinations = [
        ("Ardupilot sentences", data[data["is_ardu"]]),
        ("NATO-SFA sentences", data[~data["is_ardu"]]),
    ]

    ardu_nato_dataset = pd.DataFrame()
    for hyperparameter, data in tqdm(
        ardu_nato_combinations, total=len(ardu_nato_combinations)
    ):
        matcher = CauseEffectPairMatcher(nlp=nlp)
        sent2pair = lambda sent: extract_cause_effect_pairs(nlp(sent), matcher)
        predictions = data["sentence"].apply(sent2pair)
        metric = create_metric(
            ground_truths=data["ground_truths"],
            predictions=predictions,
        )
        metric["hyperparameter"] = hyperparameter
        ardu_nato_dataset = pd.concat([ardu_nato_dataset, metric])

    return ardu_nato_dataset


def create_annotator_agreement_dataset(validations: List[pd.DataFrame]) -> pd.DataFrame:
    validation_rb, validation_ez = validations

    metric_rb = create_metric(
        ground_truths=validation_rb["ground_truths"],
        predictions=validation_ez["ground_truths"],
    )
    metric_ez = create_metric(
        ground_truths=validation_ez["ground_truths"],
        predictions=validation_rb["ground_truths"],
    )
    data = pd.concat([metric_rb, metric_ez])
    data = data.groupby(["threshold"]).mean().reset_index()
    data["hyperparameter"] = "Annotator Agreement"
    return data


def create_metric_plot(
    data: pd.DataFrame,
    metric: str,
    colors: List[str],
    ylabel: Optional[str] = None,
) -> Figure:
    fig = plt.figure(figsize=(14, 9))
    plt.xlabel("Pair Detection Threshold (PDT)", fontsize=18)
    plt.ylabel(ylabel if ylabel else metric.capitalize(), fontsize=18)
    ax = fig.add_subplot()
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)

    for index, pattern in enumerate(data["hyperparameter"].unique()):
        ax.plot(
            "threshold",
            metric,
            data=data[data["hyperparameter"] == pattern],
            color=colors[index],
            label=pattern,
        )
        ax.legend(prop=dict(size=12))

    return fig


def take_mean_annotators(
    validations: List[pd.DataFrame],
    fn: Callable[..., pd.DataFrame],
) -> pd.DataFrame:
    data = pd.concat([fn(df) for df in validations])
    return data.groupby(["hyperparameter", "threshold"]).mean().reset_index()


def generate_precision_recall_plots() -> None:
    sns.set_theme(style="whitegrid")

    validation_cira = load_validation(Annotator.Cira)

    validations = [
        load_validation(Annotator.EhsanZibaei),
        load_validation(Annotator.RobinBorth),
    ]

    # Create and save pattern dataset
    pattern_dataset = take_mean_annotators(validations, create_pattern_dataset)
    precision_patterns = create_metric_plot(
        data=pattern_dataset,
        metric="precision",
        colors=[
            "#ff5c57",
            "#ffa057",
            "#ffdb57",
            "#b6ff57",
            "#6aff57",
            "#57f9ff",
            "#577bff",
        ],
    )
    save_plot(precision_patterns, "precision_patterns.pdf")
    recall_patterns = create_metric_plot(
        data=pattern_dataset,
        metric="recall",
        colors=[
            "#ff5c57",
            "#ffa057",
            "#ffdb57",
            "#b6ff57",
            "#6aff57",
            "#57f9ff",
            "#577bff",
        ],
    )
    save_plot(recall_patterns, "recall_patterns.pdf")

    # Create and save model dataset
    model_dataset = take_mean_annotators(validations, create_model_dataset)
    precision_models = create_metric_plot(
        data=model_dataset,
        metric="precision",
        colors=[
            "#ff5c57",
            "#b6ff57",
            "#577bff",
        ],
    )
    save_plot(precision_models, "precision_models.pdf")
    recall_models = create_metric_plot(
        data=model_dataset,
        metric="recall",
        colors=[
            "#ff5c57",
            "#b6ff57",
            "#577bff",
        ],
    )
    save_plot(recall_models, "recall_models.pdf")

    # Create and save ardu_nato dataset
    ardu_nato_dataset = take_mean_annotators(validations, create_ardu_nato_dataset)
    precision_ardu_nato = create_metric_plot(
        data=ardu_nato_dataset,
        metric="precision",
        colors=[
            "#ff5c57",
            "#577bff",
        ],
    )
    save_plot(precision_ardu_nato, "precision_ardu_nato.pdf")
    recall_ardu_nato = create_metric_plot(
        data=ardu_nato_dataset,
        metric="recall",
        colors=[
            "#ff5c57",
            "#577bff",
        ],
    )
    save_plot(recall_ardu_nato, "recall_ardu_nato.pdf")

    # Create and save the cira datset
    cira_dataset = create_ardu_nato_dataset(validation_cira)
    precision_cira = create_metric_plot(
        data=cira_dataset,
        metric="precision",
        colors=[
            "#ff5c57",
            "#577bff",
        ],
    )
    save_plot(precision_cira, "precision_cira.pdf")
    recall_cira = create_metric_plot(
        data=cira_dataset,
        metric="recall",
        colors=[
            "#ff5c57",
            "#577bff",
        ],
    )
    save_plot(recall_cira, "recall_cira.pdf")

    our_tool = ardu_nato_dataset[ardu_nato_dataset['hyperparameter'] == 'ArduCE dataset']
    cira = cira_dataset[cira_dataset['hyperparameter'] == 'ArduCE dataset']
    our_tool['hyperparameter'] = 'Our hybrid extraction based on simple-phrasal-passive pattern'
    cira['hyperparameter'] = 'Cira'
    our_tool_vs_cira = pd.concat([our_tool, cira])

    precision_fig = create_metric_plot(
        data=our_tool_vs_cira,
        metric="precision",
        colors=[
            "#ff5c57",
            "#577bff",
        ],
    )
    save_plot(precision_fig, "precision_two_tools.pdf")

    recall_fig = create_metric_plot(
        data=our_tool_vs_cira,
        metric="recall",
        colors=[
            "#ff5c57",
            "#577bff",
        ],
    )
    save_plot(recall_fig, "recall_two_tools.pdf")
    precision_average_difference = (our_tool['precision'] - cira['precision']).mean()
    recall_average_difference = (our_tool['recall'] - cira['recall']).mean()


    # Create and save annotator_agreement_dataset dataset
    annotator_agreement_dataset = create_annotator_agreement_dataset(validations)
    annotator_agreement = create_metric_plot(
        data=annotator_agreement_dataset,
        metric="precision",
        colors=["#577bff"],
        ylabel="Agreement Score",
    )
    save_plot(annotator_agreement, "annotator_agreement.pdf")


if __name__ == "__main__":
    generate_precision_recall_plots()
