from src.utils import Annotator, load_validation, save_report


def generate_validation_report() -> None:
    validation_rb = load_validation(Annotator.RobinBorth)
    validation_ez = load_validation(Annotator.EhsanZibaei)

    no_pair_sentences = (
        lambda data: data["ground_truths"]
        .apply(lambda pairs: int(len(pairs) == 0))
        .sum()
    )

    single_pair_sentences = (
        lambda data: data["ground_truths"]
        .apply(lambda pairs: int(len(pairs) == 1))
        .sum()
    )

    multiple_pair_sentence = (
        lambda data: data["ground_truths"]
        .apply(lambda pairs: int(len(pairs) > 1))
        .sum()
    )

    text = f"""
    Validation Dataset by Robin Borth:
    Total sentences: {len(validation_rb)}
    None pairs: {no_pair_sentences(validation_rb)}
    Single pair: {single_pair_sentences(validation_rb)}
    Multiple pairs: {multiple_pair_sentence(validation_rb)}

    Validation Dataset by Ehsan Zibaei:
    Total sentences: {len(validation_ez)}
    None pairs: {no_pair_sentences(validation_ez)}
    Single pair: {single_pair_sentences(validation_ez)}
    Multiple pairs: {multiple_pair_sentence(validation_ez)}
    """
    save_report(text, "validation.txt")


if __name__ == "__main__":
    generate_validation_report()
