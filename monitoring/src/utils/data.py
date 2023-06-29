import json
from typing import List, Text

import pandas as pd


def load_reference_data(columns: List[Text]) -> pd.DataFrame:
    REFERECE_PATH = (
        "/home/konradballegro/monitoring/data/reference/offers_reference.csv"
    )
    reference_data = pd.read_csv(REFERECE_PATH)
    return reference_data.loc[:, columns]


def load_current_data():
    pass
