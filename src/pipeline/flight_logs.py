import os
from itertools import groupby

import pandas as pd

from src.utils import BASE_PATH, save_data

####################################################################################################
# Dictionary from the open-source codebase from ArduPilot that is used to map the integer values
# from the historical flight logs dataset into natural language
####################################################################################################

LOG_ERROR_SUBSYSTEM = {
    "1": "main",
    "2": "radio",
    "3": "compass",
    "4": "optflow",
    "5": "failsafe radio",
    "6": "failsafe batt",
    "7": "failsafe gps",
    "8": "failsafe gcs",
    "9": "failsafe fence",
    "10": "flight mode",
    "11": "gps",
    "12": "crash check",
    "13": "flip",
    "14": "autotune",
    "15": "parachutes",
    "16": "ekf check",
    "17": "failsafe ekfinav",
    "18": "baro",
    "19": "cpu",
    "20": "failsafe adsb",
    "21": "terrain",
    "22": "navigation",
    "23": "failsafe terrain",
    "24": "ekf primary",
    "25": "thrust loss check",
    "26": "failsafe sensors",
    "27": "failsafe leak",
    "28": "pilot input",
    "29": "failsafe vibe",
    "30": "internal error",
}

LOG_EVENT_SUBSYSTEM = {
    "10": "armed",
    "11": "disarmed",
    "15": "auto armed",
    "17": "land complete maybe",
    "18": "land complete",
    "28": "not landed",
    "19": "lost gps",
    "21": "flip start",
    "22": "flip end",
    "25": "set home",
    "26": "set simple on",
    "27": "set simple off",
    "29": "set supersimple on",
    "30": "autotune initialised",
    "31": "autotune off",
    "32": "autotune restart",
    "33": "autotune success",
    "34": "autotune failed",
    "35": "autotune reached limit",
    "36": "autotune pilot testing",
    "37": "autotune savedgains",
    "38": "save trim",
    "39": "savewp add WP",
    "41": "fence enable",
    "42": "fence disable",
    "43": "acro trainer off",
    "44": "acro trainer leveling",
    "45": "acro trainer limited",
    "46": "gripper grab",
    "47": "gripper release",
    "49": "parachute disabled",
    "50": "parachute enabled",
    "51": "parachute released",
    "52": "landing gear deployed",
    "53": "landing gear retracted",
    "54": "motors emergency stopped",
    "55": "motors emergency stop cleared",
    "56": "motors interlock disabled",
    "57": "motors interlock enabled",
    "58": "rotor runup complete",
    "59": "rotor speed below critical",
    "60": "ekf alt reset",
    "61": "land cancelled by pilot",
    "62": "ekf yaw reset",
    "63": "avoidance adsb enable",
    "64": "avoidance adsb disable",
    "65": "avoidance proximity enable",
    "66": "avoidance proximity disable",
    "67": "gps primary changed",
    "71": "zigzag store a",
    "72": "zigzag store b",
    "73": "land repo active",
    "74": "standby enable",
    "75": "standby disable",
    "80": "fence floor enable",
    "81": "fence floor disable",
    "85": "ek3 sources set to primary",
    "86": "ek3 sources set to secondary",
    "87": "ek3 sources set to tertiary",
    "163": "surfaced",
    "164": "not surfaced",
    "165": "bottomed",
    "166": "not bottomed",
}


####################################################################################################
# Helper functionality to work with the pickle folder for the flight logs
####################################################################################################


def generate_dataframes():
    """Returns all dataframes from one pickle file with the corresponding file name"""
    path = BASE_PATH / "data/pipeline/00_flight_logs/raw"
    for file_name in os.listdir(path):
        if os.path.getsize(path / file_name) == 0:
            continue
        yield pd.read_pickle(path / file_name), file_name


def grab_dataframe(dataframes, tablename):
    return next((df for table, df in dataframes.items() if table == tablename), None)


####################################################################################################
# Create the a json file that includes the table and column names from the flight logs dataset
####################################################################################################


def prepare_table_column_json():
    dataframes_information = []
    for dataframes, file in generate_dataframes():
        for table, df in dataframes.items():
            if isinstance(df, pd.DataFrame):
                dataframes_information.append(
                    {
                        "table": str(table),
                        "columns": [str(column) for column in df.columns],
                    }
                )

    minified_information = []
    unique_table_names = set([data["table"] for data in dataframes_information])
    for table in unique_table_names:
        dataframes_with_same_table_name = filter(
            lambda x: x["table"] == table, dataframes_information
        )
        unique_columns = set()
        for dataframe in dataframes_with_same_table_name:
            for column in dataframe["columns"]:
                unique_columns.add(column)
        minified_information.append(
            {
                "table": table,
                "columns": list(unique_columns),
            }
        )
    return minified_information


####################################################################################################
# Create the flight logs graph for the flight logs dataset, only focusing on the error and event
# subsystem
####################################################################################################


def next_cause_effects_pairs(chain_as_text):
    # The previous chain item causes the current chain item, thus our cause effect pairs are:
    chain = chain_as_text.split("|=>")
    ceps = []
    for i, base_item in enumerate(chain):
        if i < len(chain) - 1:
            ceps.append((base_item, chain[i + 1]))
    return list(set(ceps))


def no_duplicate_neighbors(chain):
    # We reduce the chain from  15|=>15|=>15|=>15|=>16|=>17|=>17|=>15=>12 to 15|=>16|=>17|=>15=>12
    return "|=>".join([c for c, _ in groupby(chain.split("|=>"))])


def create_graph(table, column, dict_subsystem):
    # Create the chain with only the id's
    logs = []
    for dataframes, file in generate_dataframes():
        df = grab_dataframe(dataframes, table)
        if df is None:
            continue
        chain = "|=>".join([str(item) for item in df[column]])
        logs.append({"id": file, "chain": chain})
    df = pd.DataFrame(logs)
    # create from the chain the cause effect pairs
    df["chain"] = df["chain"].apply(no_duplicate_neighbors)
    cause_effect_pairs = []
    for index, row in df.iterrows():
        ceps = next_cause_effects_pairs(row["chain"])
        for cause, effect in ceps:
            cause_effect_pairs.append(
                {"id": row["id"], "cause": cause, "effect": effect}
            )
    df = pd.DataFrame(cause_effect_pairs)

    # The logs only have the actual error or event stored as integer, we now need to map the int to the words
    df["cause"] = df["cause"].apply(lambda i: dict_subsystem.get(i, ""))
    df["effect"] = df["effect"].apply(lambda i: dict_subsystem.get(i, ""))
    # Removes the rows that are potentially destroyed during prep
    df = df[(df["cause"] != "") & (df["effect"] != "")]
    # Sum up the unique cause effect pairs
    df = df.groupby(["cause", "effect"]).size().reset_index(name="weight")
    return df


def main():
    error_logs = create_graph(
        table="ERR",
        column="Subsys",
        dict_subsystem=LOG_ERROR_SUBSYSTEM,
    )
    event_logs = create_graph(
        table="EV",
        column="Id",
        dict_subsystem=LOG_EVENT_SUBSYSTEM,
    )
    flight_logs = pd.concat([error_logs, event_logs])
    flight_logs = flight_logs.groupby(["cause", "effect"]).sum().reset_index()
    save_data(flight_logs, "pipeline/00_flight_logs/flight_logs.csv")


if __name__ == "__main__":
    main()
