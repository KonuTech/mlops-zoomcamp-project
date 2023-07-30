"""
Flask API for Used Car Price Prediction

This script provides a Flask API for predicting used car prices.
It reads JSON data containing car offers,
filters the data based on specified conditions,
preprocesses the data by cleaning and transforming columns,
performs feature engineering to create additional features,
and uses a trained machine learning model to predict the prices.

Modules:
    - json: Handles JSON data.
    - logging: Configures and performs logging.
    - os: Provides interface with the operating system.
    - functools: Used for reducing the conditions list.
    - typing.List: Represents a list type hint.
    - mlflow: Integrates with MLflow for model tracking and loading.
    - numpy as np: Used for numerical operations.
    - pandas as pd: Provides data manipulation capabilities.
    - flask.Flask: Creates a Flask web application.
    - flask.jsonify: Creates a JSON response for the Flask API.
    - flask.request: Handles incoming HTTP requests in Flask.
    - werkzeug.exceptions.HTTPException: Represents an HTTP exception in Flask.

Constants:
    - CONFIG_PATH: Path to the configuration file containing settings.
    - DISTINCT_COLUMNS: List of columns for feature engineering.
    - COLUMNS_TO_DROP: List of columns to drop from the DataFrame.
    - COLUMNS_TO_ADD: List of columns to add to the DataFrame if they don't exist.
    - SELECTED_FEATURES: List of columns to select as the final set of features.
    - RUN_ID: MLflow run ID of the trained machine learning model.
    - logged_model: GCS path to the trained machine learning model.
    - model: Loaded MLflow model for making predictions.

Functions:
    - data_filter: Filters the data based on specified conditions.
    - data_preprocess: Preprocesses the data by cleaning and transforming columns.
    - features_engineer: Performs feature engineering to create additional features.
    - prepare_features: Prepares features by reading, filtering, preprocessing,
    and performing feature engineering.
    - predict_endpoint: Flask endpoint for making predictions.

Note:
    This script assumes that the trained machine learning model
    has been logged using MLflow
    and the model URI is specified in the `logged_model` variable.
"""

import json
import logging
import os
from functools import reduce
from typing import List

import mlflow
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from werkzeug.exceptions import HTTPException

CONFIG_PATH = "/home/konradballegro/scoring_batch/config/config.json"


with open(CONFIG_PATH, encoding="UTF-8") as json_file:
    config = json.load(json_file)


# FILE_PATH = config["FILE_PATH"]
DISTINCT_COLUMNS = config["DISTINCT_COLUMNS"]
COLUMNS_TO_DROP = config["COLUMNS_TO_DROP"]
COLUMNS_TO_ADD = config["COLUMNS_TO_ADD"]
SELECTED_FEATURES = config["SELECTED_FEATURES"]


# TRACKING_SERVER_HOST = "34.77.180.77"
RUN_ID = "12e03b0d8db04dbe99467a2bcde74183"

# Configure logging
logging.basicConfig(
    filename="/home/konradballegro/scoring_batch/app.log",
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


# Configure MLflow
logged_model = f"gs://mlops-zoomcamp/3/{RUN_ID}/artifacts/model"
model = mlflow.pyfunc.load_model(logged_model)

logging.debug("Model loaded: %s", logged_model)


app = Flask("otomoto-used-car-price-prediction")


def data_filter(json_string: str) -> pd.DataFrame:
    """
    Filters the data based on specified conditions.

    Args:
        json_string  (str): The input JSON string.

    Returns:
        filtered_data_frame (pd.DataFrame): The filtered DataFrame.
    """
    logging.info("Read JSON string...")
    data_frame = pd.read_json(json_string, orient="records")

    # Log the DataFrame
    logging.warning("DataFrame type: %s", type(data_frame))
    logging.warning("Filtering data...")
    conditions = [
        data_frame["Currency"] == "PLN",
        data_frame["Country of origin"] == "Polska",
        data_frame["Accident-free"].notnull(),
        data_frame["Price"].notnull(),
        data_frame["Offer from"].notnull(),
        data_frame["Condition"].notnull(),
        data_frame["Condition"] == "Używane",
        data_frame["Vehicle brand"].notnull(),
        data_frame["Year of production"].notnull(),
        data_frame["Mileage"].notnull(),
        data_frame["Fuel type"].notnull(),
        data_frame["Power"].notnull(),
        data_frame["Gearbox"].notnull(),
        data_frame["Body type"].notnull(),
        data_frame["Number of doors"].notnull(),
    ]
    filtered_data_frame = data_frame.loc[reduce(lambda a, b: a & b, conditions), :]
    logging.warning("Data filtered")
    return filtered_data_frame


def data_preprocess(data_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the data by cleaning and transforming the columns.

    Args:
        data_frame (pd.DataFrame): The input Pandas DataFrame.

    Returns:
        cleaned_data_frame (pd.DataFrame): The preprocessed Pandas DataFrame.

    """
    logging.warning("Preprocessing data...")
    data_frame_cleaned = pd.DataFrame()
    data_frame_cleaned["Price"] = data_frame["Price"].apply(
        lambda x: int(float(x.replace(",", ""))) if isinstance(x, str) else int(x)
    )
    data_frame_cleaned["Offer from"] = data_frame["Offer from"]
    data_frame_cleaned["Condition"] = data_frame["Condition"]
    data_frame_cleaned["Vehicle brand"] = data_frame["Vehicle brand"]
    data_frame_cleaned["Vehicle model"] = data_frame["Vehicle model"]
    data_frame_cleaned["Year of production"] = data_frame["Year of production"].astype(
        str
    )
    data_frame_cleaned["Mileage"] = (
        data_frame["Mileage"]
        .astype(str)
        .str.replace(" ", "")
        .str.replace("km", "")
        .astype(np.float32)  # Convert to float32
    )
    data_frame_cleaned["Fuel type"] = data_frame["Fuel type"]
    data_frame_cleaned["Power"] = (
        data_frame["Power"]
        .astype(str)
        .str.replace(" ", "")
        .str.replace("KM", "")
        .astype(np.int32)  # Convert to int32
    )
    data_frame_cleaned["Gearbox"] = data_frame["Gearbox"]
    data_frame_cleaned["Body type"] = data_frame["Body type"]
    data_frame_cleaned["Number of doors"] = (
        data_frame["Number of doors"].astype(str).str.replace(r"\.0$", "", regex=True)
    )
    data_frame_cleaned["URL path"] = data_frame["URL path"]
    data_frame_cleaned["ID"] = data_frame["ID"]
    data_frame_cleaned["Epoch"] = data_frame["Epoch"]

    data_frame.to_csv(
        "/home/konradballegro/data/preprocessed/offers_filtered.csv", index=False
    )

    logging.warning("Data preprocessed")
    return data_frame_cleaned


def features_engineer(
    data_frame: pd.DataFrame,
    distinct_columns: List[str],
    columns_to_drop: List[str],
    columns_to_add: List[str],
    selected_features: List[str],
) -> pd.DataFrame:
    """
    Performs feature engineering to create additional features based on the input DataFrame.

    Args:
        data_frame (pd.DataFrame): The input Pandas DataFrame.
        distinct_columns (list): A list of columns to perform feature engineering.
        columns_to_drop (list): A list of columns to drop from the DataFrame.
        columns_to_add (List[str]): A list of columns to add to the DataFrame if they don't exist.
        selected_features (List[str]): A list of columns to select as the final set of features.

    Returns:
        features_data_frame (np.ndarray): NumPy array with additional engineered features.

    """
    logging.warning("Performing feature engineering...")
    # Iterate over distinct_columns
    for column in distinct_columns:
        # Get distinct values for the column
        distinct_values = data_frame[column].unique()

        # Iterate over distinct values
        for value in distinct_values:
            # Create a new column name based on distinct value
            column_name = f"{column.replace(' ', '_')}_{value.replace(' ', '_')}"

            # Create a dummy variable for the distinct value
            dummy_variable = (data_frame[column] == value).astype(int)
            # dummy_variable = (data_frame[column].astype(str) == str(value)).astype(int)

            # Assign the dummy variable to the new column
            data_frame[column_name] = dummy_variable

    # Check if each column exists in the DataFrame
    columns_to_add = [col for col in columns_to_add if col not in data_frame.columns]

    # Add the missing columns with default value 0 using pd.concat()
    data_frame = pd.concat(
        [data_frame, pd.DataFrame(0, index=data_frame.index, columns=columns_to_add)],
        axis=1,
    )
    data_frame = data_frame.drop(columns_to_drop, axis=1)
    data_frame = data_frame.dropna(subset=["Price"])

    # Iterate over each feature in SELECTED_FEATURES
    for feature in SELECTED_FEATURES:
        if feature not in ["Mileage", "Power"]:
            if feature in data_frame.columns:
                if data_frame[feature].dtype != np.int32:
                    data_frame[feature] = data_frame[feature].astype(np.int32)

    data_frame = data_frame[selected_features]
    data_frame.to_csv(
        "/home/konradballegro/data/preprocessed/offers_preprocessed.csv", index=False
    )
    # Log the DataFrame
    logging.warning("DataFrame type: %s", type(data_frame))
    # Get the shape of the DataFrame
    num_rows, num_cols = data_frame.shape
    logging.warning("Number of rows:%s", num_rows)
    logging.warning("Number of columns:%s", num_cols)
    logging.warning("DataFrame:\n%s", data_frame)
    logging.warning("Feature engineering completed")

    return data_frame


def prepare_features(json_string: str) -> pd.DataFrame:
    """
    Prepares features by reading, filtering, preprocessing, and performing feature engineering.

    Args:
        json_string (str): JSON string containing the data.

    Returns:
        features (np.ndarray): NumPy array with prepared features.
    """

    # Filter data
    data_filtered = data_filter(json_string=json_string)

    # Preprocess data
    data_preprocessed = data_preprocess(data_frame=data_filtered)

    # Perform feature engineering
    features = features_engineer(
        data_frame=data_preprocessed,
        distinct_columns=DISTINCT_COLUMNS,
        columns_to_drop=COLUMNS_TO_DROP,
        columns_to_add=COLUMNS_TO_ADD,
        selected_features=SELECTED_FEATURES,
    )

    return features


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    """
    Flask endpoint for making predictions.

    Accepts a POST request with a JSON payload containing data.
    Prepares the features, makes predictions using the trained model,
    and returns the predictions as a JSON response.

    Returns:
        JSON response with predictions and model version.
    """
    try:
        json_string = request.get_json()
        logging.debug("Received JSON data: %s", json_string)

        features = prepare_features(json_string=json_string)
        predictions = model.predict(features)  # Pass the DataFrame as input

        result = {"price": predictions.tolist(), "model_version": RUN_ID}

        # Concatenate features and predictions
        # data_with_predictions = pd.concat(
        #     [features, pd.DataFrame(predictions, columns=["predictions"])], axis=1
        # )

        # Load existing offers.csv file if it exists
        if os.path.isfile("/home/konradballegro/data/scored/offers_scored.csv"):
            existing_data = pd.read_csv(
                "/home/konradballegro/data/preprocessed/offers_filtered.csv"
            )
            output = pd.concat(
                [existing_data, pd.DataFrame(predictions, columns=["predictions"])],
                axis=1,
            )
            existing_preprocessed_data = pd.read_csv(
                "/home/konradballegro/data/preprocessed/offers_preprocessed.csv"
            )
            output_current = pd.concat(
                [
                    existing_preprocessed_data,
                    pd.DataFrame(predictions, columns=["predictions"]),
                    existing_data["Price"],
                ],
                axis=1,
            )

        # Save the concatenated data as offers.csv
        output.to_csv("/home/konradballegro/data/scored/offers_scored.csv", index=False)
        output_current.to_csv(
            "/home/konradballegro/data/scored/offers_scored_current.csv", index=False
        )

        return jsonify(result)
    except HTTPException as error:
        logging.exception("HTTPException occurred")
        return jsonify({"error": str(error)}), error.code


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=9696)
