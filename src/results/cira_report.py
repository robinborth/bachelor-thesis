#%%
from src.utils import Annotator, load_validation

validation = load_validation(Annotator.Cira)
validation["ground_truths"].apply(len)

#%%


def generate_cira_report() -> None:
    validation = load_validation(Annotator.Cira)
    validation["ground_truths"].apply(len)


if __name__ == "__main__":
    generate_cira_report()
