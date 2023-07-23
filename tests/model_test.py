import sys
import json
import numpy as np
import pandas as pd
import mlflow

# Add the desired path to sys.path
sys.path.append("/home/konradballegro/scoring_batch")
from app import data_filter, data_preprocess, features_engineer


CONFIG_PATH = "/home/konradballegro/tests/config/config.json"

with open(CONFIG_PATH, encoding="UTF-8") as json_file:
    config = json.load(json_file)

FILE_PATH = config["FILE_PATH"]
SCORED_FILE_PATH = config["SCORED_FILE_PATH"]
DISTINCT_COLUMNS = config["DISTINCT_COLUMNS"]
COLUMNS_TO_DROP = config["COLUMNS_TO_DROP"]
COLUMNS_TO_ADD = config["COLUMNS_TO_ADD"]
SELECTED_FEATURES = config["SELECTED_FEATURES"]
DATA_TYPES = config["DATA_TYPES"]

RUN_ID = "12e03b0d8db04dbe99467a2bcde74183"
logged_model = f"gs://mlops-zoomcamp/3/{RUN_ID}/artifacts/model"
model = mlflow.pyfunc.load_model(logged_model)


def data_read(file_path: str) -> str:
    """
    Reads the data from the specified file.
    Args:
        file_path (str): Path to the file to be read.
    Returns:
        data_json (str): JSON string containing the read data.
    """
    df = pd.read_csv(file_path, index_col=False, low_memory=False)
    data_json = df.to_json()
    return data_json


def test_prepare_features():
    actual_features = data_read(file_path=FILE_PATH)
    actual_features_filtered = data_filter(actual_features)
    actual_features_preprocessed = data_preprocess(actual_features_filtered)
    actual_features_engineered = features_engineer(
        df=actual_features_preprocessed,
        distinct_columns=DISTINCT_COLUMNS,
        columns_to_drop=COLUMNS_TO_DROP,
        columns_to_add=COLUMNS_TO_ADD,
        selected_features=SELECTED_FEATURES
    )

    actual_features_engineered.to_csv(
        "/home/konradballegro/tests/data/preprocessed/nissan_preprocessed.csv", index=False
    )

    actual_features = pd.read_csv("/home/konradballegro/tests/data/preprocessed/nissan_preprocessed.csv")
    expected_features = pd.read_csv("/home/konradballegro/tests/data/preprocessed/offers_preprocessed.csv")

    assert actual_features.equals(expected_features)


def test_prediction():
    last_column_index = -1
    prediction_reference = pd.read_csv(SCORED_FILE_PATH, usecols=['predictions'], dtype='float32')
    print(prediction_reference.values[-1][0])
    print(type(prediction_reference.values[-1][0]))

    # Define the data types for each column
    data_types = DATA_TYPES
    
    actual_features = pd.read_csv("/home/konradballegro/tests/data/preprocessed/nissan_preprocessed.csv", dtype=data_types)
    prediction_got = model.predict(actual_features)
    print(prediction_got[-1])
    print(type(prediction_got[-1]))
    
    assert np.array_equal(prediction_got[-1], prediction_reference.values[-1][0])
