from src.utils import (
    load_cause_effect_pairs,
    load_prefiltered,
    load_source,
    save_report,
)


def generate_distribution_report() -> None:
    source = load_source()
    prefiltered = load_prefiltered()
    pairs = load_cause_effect_pairs()

    sentence_count = lambda data, name: len(data[data["source"] == name])

    text = f"""
    02 Source
    Total sentences: {len(source)}
    Blog sentences: {sentence_count(source, "blog")}
    Docs sentences: {sentence_count(source, "docs")}
    Discord sentences: {sentence_count(source, "discord")}

    03 Prefiltered
    Total sentences: {len(prefiltered)}
    Blog sentences: {sentence_count(prefiltered, "blog")}
    Docs sentences: {sentence_count(prefiltered, "docs")}
    Discord sentences: {sentence_count(prefiltered, "discord")}

    04 Cause-Effect-Pairs
    Total sentences: {len(pairs)}
    Blog sentences: {sentence_count(pairs, "blog")}
    Docs sentences: {sentence_count(pairs, "docs")}
    Discord sentences: {sentence_count(pairs, "discord")}
    """
    save_report(text, "distribution.txt")


if __name__ == "__main__":
    generate_distribution_report()
