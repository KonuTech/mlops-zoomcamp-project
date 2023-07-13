import os
import json
import logging
from functools import reduce
from typing import List

import mlflow
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request

CONFIG_PATH = "/home/konradballegro/scoring-batch/config/config.json"


with open(CONFIG_PATH, encoding="UTF-8") as json_file:
    config = json.load(json_file)


FILE_PATH = config["FILE_PATH"]
DISTINCT_COLUMNS = config["DISTINCT_COLUMNS"]
COLUMNS_TO_DROP = config["COLUMNS_TO_DROP"]
COLUMNS_TO_ADD = config["COLUMNS_TO_ADD"]
SELECTED_FEATURES = config["SELECTED_FEATURES"]


# TRACKING_SERVER_HOST = "34.77.180.77"
# RUN_ID = "ed549f18f6a64334b9873babbcb43dee"
RUN_ID = "12e03b0d8db04dbe99467a2bcde74183"

# Configure logging
logging.basicConfig(
    filename="/home/konradballegro/scoring-batch/app.log",
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


# Configure MLflow
# mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:5000")
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
        filtered_df (pd.DataFrame): The filtered DataFrame.
    """
    logging.info("Read JSON string...")
    df = pd.read_json(json_string)
    
    # Log the DataFrame
    logging.warning("DataFrame type: %s", type(df))
    # Get the shape of the DataFrame
    num_rows, num_cols = df.shape
    logging.warning("Number of rows:%s", num_rows)
    logging.warning("Number of columns:%s", num_cols)
    logging.warning("DataFrame:\n%s", df)


    logging.warning("Filtering data...")
    conditions = [
        df["Currency"] == "PLN",
        df["Country of origin"] == "Polska",
        df["Accident-free"].notnull(),
        df["Price"].notnull(),
        df["Offer from"].notnull(),
        df["Condition"].notnull(),
        df["Condition"] == "Używane",
        df["Vehicle brand"].notnull(),
        df["Year of production"].notnull(),
        df["Mileage"].notnull(),
        df["Fuel type"].notnull(),
        df["Power"].notnull(),
        df["Gearbox"].notnull(),
        df["Body type"].notnull(),
        df["Number of doors"].notnull(),
    ]
    filtered_df = df.loc[reduce(lambda a, b: a & b, conditions), :]
    logging.warning("Data filtered")
    return filtered_df


def data_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the data by cleaning and transforming the columns.

    Args:
        df (pd.DataFrame): The input Pandas DataFrame.

    Returns:
        cleaned_df (pd.DataFrame): The preprocessed Pandas DataFrame.

    """
    logging.warning("Preprocessing data...")
    df_cleaned = pd.DataFrame()
    df_cleaned["Price"] = df["Price"].astype(float)
    df_cleaned["Offer from"] = df["Offer from"]
    df_cleaned["Condition"] = df["Condition"]
    df_cleaned["Vehicle brand"] = df["Vehicle brand"]
    df_cleaned["Vehicle model"] = df["Vehicle model"]
    df_cleaned["Year of production"] = df["Year of production"].astype(str)
    df_cleaned["Mileage"] = (
        df["Mileage"]
        .astype(str)
        .str.replace(" ", "")
        .str.replace("km", "")
        .astype(np.float32)  # Convert to float32
    )
    df_cleaned["Fuel type"] = df["Fuel type"]
    df_cleaned["Power"] = (
        df["Power"].astype(str).str.replace(" ", "").str.replace("KM", "").astype(np.int32)  # Convert to int32
    )
    df_cleaned["Gearbox"] = df["Gearbox"]
    df_cleaned["Body type"] = df["Body type"]
    df_cleaned["Number of doors"] = (df["Number of doors"].astype(str).str.replace(r'\.0$', ""))
    df_cleaned["URL path"] = df["URL path"]
    df_cleaned["ID"] = df["ID"]
    df_cleaned["Epoch"] = df["Epoch"]

    df.to_csv(
        "/home/konradballegro/data/preprocessed/offers_filtered.csv", index=False
    )

    logging.warning("Data preprocessed")
    return df_cleaned


def features_engineer(
    df: pd.DataFrame,
    distinct_columns: List[str],
    columns_to_drop: List[str],
    columns_to_add: List[str],
    selected_features: List[str]
    ) -> pd.DataFrame:
    """
    Performs feature engineering to create additional features based on the input DataFrame.

    Args:
        df (pd.DataFrame): The input Pandas DataFrame.
        distinct_columns (list): A list of columns to perform feature engineering on distinct values.
        columns_to_drop (list): A list of columns to drop from the DataFrame.
        columns_to_add (List[str]): A list of columns to add to the DataFrame if they don't exist.
        selected_features (List[str]): A list of columns to select as the final set of features.

    Returns:
        features_df (np.ndarray): NumPy array with additional engineered features.

    """
    logging.warning("Performing feature engineering...")
    # Iterate over distinct_columns
    for column in distinct_columns:
        # Get distinct values for the column
        distinct_values = df[column].unique()

        # Iterate over distinct values
        for value in distinct_values:
            # Create a new column name based on distinct value
            column_name = f"{column.replace(' ', '_')}_{value.replace(' ', '_')}"

            # Create a dummy variable for the distinct value
            dummy_variable = (df[column] == value).astype(int)
            # dummy_variable = (df[column].astype(str) == str(value)).astype(int)

            # Assign the dummy variable to the new column
            df[column_name] = dummy_variable

    # Check if each column exists in the DataFrame
    columns_to_add = [col for col in columns_to_add if col not in df.columns]

    # Add the missing columns with default value 0 using pd.concat()
    df = pd.concat([df, pd.DataFrame(0, index=df.index, columns=columns_to_add)], axis=1)
    df = df.drop(columns_to_drop, axis=1)
    df = df.dropna(subset=["Price"])

   # Iterate over each feature in SELECTED_FEATURES
    for feature in SELECTED_FEATURES:
        if feature not in ["Mileage", "Power"]:
            if feature in df.columns:
                if df[feature].dtype != np.int32:
                    df[feature] = df[feature].astype(np.int32)

    df = df[selected_features]
    df.to_csv(
        "/home/konradballegro/data/preprocessed/offers_preprocessed.csv", index=False
    )
    # Log the DataFrame
    logging.warning("DataFrame type: %s", type(df))
    # Get the shape of the DataFrame
    num_rows, num_cols = df.shape
    logging.warning("Number of rows:%s", num_rows)
    logging.warning("Number of columns:%s", num_cols)
    logging.warning("DataFrame:\n%s", df)
    logging.warning("Feature engineering completed")

    return df


def prepare_features(json_string: str) -> pd.DataFrame:
    """
    Prepares features by reading, filtering, preprocessing, and performing feature engineering on the data.

    Args:
        json_string (str): JSON string containing the data.

    Returns:
        features (np.ndarray): NumPy array with prepared features.
    """

    # Filter data
    data_filtered = data_filter(json_string=json_string)

    # Preprocess data
    data_preprocessed = data_preprocess(df=data_filtered)

    # Perform feature engineering
    features = features_engineer(
        df=data_preprocessed,
        distinct_columns=DISTINCT_COLUMNS,
        columns_to_drop=COLUMNS_TO_DROP,
        columns_to_add=COLUMNS_TO_ADD,
        selected_features=SELECTED_FEATURES
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
        data_with_predictions = pd.concat([features, pd.DataFrame(predictions, columns=["predictions"])], axis=1)
        # data_with_predictions = pd.merge(features, pd.DataFrame(predictions, columns=["predictions"]), left_index=True, right_index=True)
        # Load existing offers.csv file if it exists
        if os.path.isfile("/home/konradballegro/data/scored/offers_scored.csv"):
            # existing_data = pd.read_csv("/home/konradballegro/data/preprocessed/offers_preprocessed.csv")
            existing_data = pd.read_csv("/home/konradballegro/data/preprocessed/offers_filtered.csv")
            output = pd.concat([existing_data, pd.DataFrame(predictions, columns=["predictions"])], axis=1)

        # Save the concatenated data as offers.csv
        output.to_csv("/home/konradballegro/data/scored/offers_scored.csv", index=False)

        return jsonify(result)
    except Exception as e:
        logging.exception("Prediction failed")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=9696)
