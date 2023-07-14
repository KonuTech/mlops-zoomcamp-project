from typing import List, Text

import pandas as pd


def load_reference_data(columns: List[Text]) -> pd.DataFrame:
    cur_path = "/home/konradballegro/data/scored/offers_scored_reference.csv"
    cur_data = pd.read_csv(cur_path)
    current_data = cur_data.loc[:, columns]
    return current_data


def load_current_data(columns: List[Text]) -> pd.DataFrame:
    ref_path = "/home/konradballegro/data/scored/offers_scored_current.csv"
    ref_data = pd.read_csv(ref_path)
    reference_data = ref_data.loc[:, columns]
    return reference_data
