import json

import pandas as pd

from src.utils import Annotator, load_raw_validation, save_data


def extract_features(val_data: pd.DataFrame) -> pd.DataFrame:
    val_data["is_ardu"] = ~val_data["foreignDomain"]
    val_data["tokens"] = val_data["tokens"].apply(json.dumps)
    val_data = val_data[(val_data["labelState"] == "accepting") & (~val_data["ignore"])]
    return val_data[["id", "sentence", "is_ardu", "tokens"]].reset_index(drop=True)


def build_datasets() -> None:
    raw_validation_rb = load_raw_validation(Annotator.RobinBorth)
    validation_rb = extract_features(raw_validation_rb)

    raw_validation_ez = load_raw_validation(Annotator.EhsanZibaei)
    validation_ez = extract_features(raw_validation_ez)

    validation_rb = validation_rb[validation_rb["id"].isin(validation_ez["id"])]
    validation_ez = validation_ez[validation_ez["id"].isin(validation_rb["id"])]

    save_data(validation_ez, f"validation/validation_{Annotator.EhsanZibaei.value}.csv")
    save_data(validation_rb, f"validation/validation_{Annotator.RobinBorth.value}.csv")


if __name__ == "__main__":
    build_datasets()
