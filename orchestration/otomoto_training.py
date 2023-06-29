"""
This script performs the training and evaluation of a machine learning model using data scraped from otomoto.pl .

The following steps are performed in this script:

1. Download the "offers.csv" file from a Google Cloud Storage bucket to a local directory.
2. Create a Spark session.
3. Read the data from the "offers.csv" file using Spark.
4. Filter the data based on specified conditions.
5. Preprocess the data by cleaning and transforming the columns.
6. Perform feature engineering to create additional features.
7. Split the data into train, validation, and test sets.
8. Perform feature selection using Recursive Feature Elimination with Cross-Validation (RFECV) and XGBoost.
9. Perform hyperparameter tuning using GridSearchCV on XGBoost.
10. Train the final model using the selected features and hyperparameters.
11. Evaluate the model on the test set.
12. Save the trained model.

Note: The script uses various configuration parameters specified in the "config.json" file.

"""
import json
import logging
import os
from functools import reduce
from typing import Dict, List, Tuple

import mlflow
import numpy as np
import pandas as pd
import pyspark
from google.cloud import storage
from prefect import flow, task
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, regexp_replace, when
from sklearn.feature_selection import RFE, RFECV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from xgboost import XGBRegressor

# Load configuration from config.json
os.environ[
    "GOOGLE_APPLICATION_CREDENTIALS"
] = "/home/konradballegro/.ssh/ny-rides-konrad-b5129a6f2e66.json"
CONFIG_PATH = "/home/konradballegro/orchestration/config/config.json"


with open(CONFIG_PATH) as json_file:
    config = json.load(json_file)


# Extracting config variables
BUCKET_NAME = config["BUCKET_NAME"]
SOURCE_PATH = config["SOURCE_PATH"]
DESTINATION_PATH = config["DESTINATION_PATH"]
FILE_NAME = config["FILE_NAME"]
SPARK_SESSION_SCOPE = config["SPARK_SESSION_SCOPE"]
SPARK_SESSION_NAME = config["SPARK_SESSION_NAME"]
HEADER = config["HEADER"]
INFER_SCHEMA = config["INFER_SCHEMA"]
FILE_PATH = config["FILE_PATH"]
FILE_PREPROCESSED_PATH = config["FILE_PREPROCESSED_PATH"]
TARGET_NAME = config["TARGET_NAME"]
TARGET_OUTPUT_DISTRIBUTION = config["TARGET_OUTPUT_DISTRIBUTION"]
DISTINCT_COLUMNS = config["DISTINCT_COLUMNS"]
COLUMNS_TO_DROP = config["COLUMNS_TO_DROP"]
DOOR_COLUMNS = config["DOOR_COLUMNS"]
BODY_COLUMNS = config["BODY_COLUMNS"]
FUEL_COLUMNS = config["FUEL_COLUMNS"]
BRAND_COLUMNS = config["BRAND_COLUMNS"]
YEAR_COLUMNS = config["YEAR_COLUMNS"]
REMAINDER_SIZE = config["REMAINDER_SIZE"]
TEST_SIZE = config["TEST_SIZE"]
RANDOM_STATE = config["RANDOM_STATE"]
MODEL_PATH = config["MODEL_PATH"]
REGRESSOR_GRID = config["REGRESSOR_GRID"]
METRIC = config["METRIC"]
X_TEST_PATH = config["X_TEST_PATH"]
Y_TEST_PATH = config["Y_TEST_PATH"]
Y_PRED_PATH = config["Y_PRED_PATH"]
CURRENT_PATH = config["CURRENT_PATH"]


# Create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")


# Create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")


# Copy offercs.csv from GCP bucket to VM
@task(retries=0, retry_delay_seconds=2)
def files_download(bucket_name: str, source_path: str, destination_path: str) -> None:
    """
    Downloads the "offers.csv" file from the specified Google Cloud Storage bucket to the local destination path.

    Args:
        bucket_name (str): The name of the Google Cloud Storage bucket.
        source_path (str): The path of the file within the bucket.
        destination_path (str): The local destination path to save the downloaded file.

    Returns:
        None

    """
    logger.info(f"Downloading file from bucket '{bucket_name}' to '{destination_path}'")
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_path)
    blob.download_to_filename(destination_path)
    logger.info("File downloaded successfully")


@task(retries=0, retry_delay_seconds=2)
def session_create(spark_session_scope: str, spark_session_name: str) -> SparkSession:
    """
    Creates a Spark session with the specified scope and name.

    Args:
        spark_session_scope (str): The Spark session scope.
        spark_session_name (str): The Spark session name.

    Returns:
        session (SparkSession): The created Spark session.

    """
    logger.info("Creating Spark session...")
    session = (
        SparkSession.builder.master(spark_session_scope)
        .appName(spark_session_name)
        .getOrCreate()
    )
    logger.info("Spark session created")
    return session


@task(retries=0, retry_delay_seconds=2)
def data_read(
    spark_session: SparkSession, header: bool, infer_schema: bool, file_path: str
) -> DataFrame:
    """
    Reads the data from the specified file using the provided Spark session.

    Args:
        spark_session (SparkSession): The Spark session.
        header (bool): Whether the file has a header row.
        infer_schema (bool): Whether to infer the schema of the data.
        file_path (str): The path of the file to read.

    Returns:
        df (DataFrame): The Spark DataFrame containing the read data.

    """
    logger.info(f"Reading data from file: '{file_path}'")
    return (
        spark_session.read.option("header", header)
        .option("inferSchema", infer_schema)
        .csv(file_path)
    )


@task(retries=0, retry_delay_seconds=2)
def data_filter(df) -> DataFrame:
    """
    Filters the data based on specified conditions.

    Args:
        df (DataFrame): The input Spark DataFrame.

    Returns:
        filtered_df (DataFrame): The filtered Spark DataFrame.

    """
    logger.info("Filtering data...")
    conditions = [
        (df["Currency"] == "PLN"),
        (df["Country of origin"] == "Polska"),
        df["Accident-free"].isNotNull(),
        df["Price"].isNotNull(),
        df["Offer from"].isNotNull(),
        df["Condition"].isNotNull(),
        (df["Condition"] == "Używane"),
        df["Vehicle brand"].isNotNull(),
        df["Year of production"].isNotNull(),
        df["Mileage"].isNotNull(),
        df["Fuel type"].isNotNull(),
        df["Power"].isNotNull(),
        df["Gearbox"].isNotNull(),
        df["Body type"].isNotNull(),
        df["Number of doors"].isNotNull(),
    ]
    logger.info("Data filtered")
    return df.filter(reduce(lambda a, b: a & b, conditions))


@task(retries=0, retry_delay_seconds=2)
def data_preprocess(df: DataFrame) -> DataFrame:
    """
    Preprocesses the data by cleaning and transforming the columns.

    Args:
        df (DataFrame): The input Spark DataFrame.

    Returns:
        cleaned_df (DataFrame): The preprocessed Spark DataFrame.

    """
    logger.info("Preprocessing data...")
    df_cleaned = df.select(
        col("Price").cast("float").alias("Price"),
        "Offer from",
        "Condition",
        "Vehicle brand",
        "Vehicle model",
        col("Year of production").cast("string").alias("Year of production"),
        regexp_replace(regexp_replace(col("Mileage"), " ", ""), "km", "")
        .cast("float")
        .alias("Mileage"),
        "Fuel type",
        regexp_replace(regexp_replace(col("Power"), " ", ""), "KM", "")
        .cast("integer")
        .alias("Power"),
        "Gearbox",
        "Body type",
        regexp_replace(col("Number of doors"), "\\.0$", "").alias("Number of doors"),
        "URL path",
        "ID",
        "Epoch",
    )
    logger.info("Data preprocessed")
    return df_cleaned


@task(retries=0, retry_delay_seconds=2)
def features_engineer(
    df: DataFrame,
    distinct_columns: List[str],
    columns_to_drop: List[str],
    brand_columns: List[str],
    fuel_columns: List[str],
    body_columns: List[str],
    door_columns: List[str],
    file_preprocessed_path: str,
) -> DataFrame:
    """
        Performs feature engineering to create additional features based on the input DataFrame.

        Args:
            df (DataFrame): The input Spark DataFrame.
            distinct_columns (list): A list of columns to perform feature engineering on distinct values.
            columns_to_drop (list): A list of columns to drop from the DataFrame.
            brand_columns (list): A list of columns representing different vehicle brands.
            fuel_columns (list): A list of columns representing different fuel types.
            body_columns (list): A list of columns representing different body types.
            door_columns (list): A list of columns representing different number of doors.
            file_preprocessed_path (str): The path to save the preprocessed offers DataFrame.

    Returns:
        features_df (DataFrame): Pandas DataFrame with additional engineered features.

    """
    logger.info("Performing feature engineering...")
    for column in distinct_columns:
        distinct_values = (
            df.select(column).distinct().rdd.flatMap(lambda x: x).collect()
        )
        for value in distinct_values:
            column_name = f"{column.replace(' ', '_')}_{value.replace(' ', '_')}"
            df = df.withColumn(column_name, when(df[column] == value, 1).otherwise(0))

    df = df.drop(*columns_to_drop).filter(df["Price"].isNotNull())

    df_pd = df.toPandas()

    # Price per Mileage
    df_pd["price_per_mileage"] = df_pd["Price"] / df_pd["Mileage"]

    # Power-to-Price Ratio
    df_pd["power_to_price_ratio"] = df_pd["Power"] / df_pd["Price"]

    # Number of Brands
    df_pd["brand_count"] = df_pd[brand_columns].sum(axis=1)
    df_pd["brand_ratio"] = df_pd[brand_columns].sum(axis=1) / len(brand_columns)

    # Fuel Type Count
    df_pd["fuel_type_count"] = df_pd[fuel_columns].sum(axis=1)
    df_pd["fuel_type_ratio"] = df_pd[fuel_columns].sum(axis=1) / len(fuel_columns)

    # Body Type Count
    df_pd["body_type_count"] = df_pd[body_columns].sum(axis=1)
    df_pd["body_type_ratio"] = df_pd[body_columns].sum(axis=1) / len(body_columns)

    # Door Count
    df_pd["door_number_count"] = df_pd[door_columns].sum(axis=1)
    df_pd["door_number_ratio"] = df_pd[door_columns].sum(axis=1) / len(door_columns)

    df_pd.to_csv(file_preprocessed_path, index=False)
    logger.info("Feature engineering completed")
    return df_pd


@task(retries=0, retry_delay_seconds=2)
def data_split(
    df: DataFrame,
    target_name: str,
    remainder_size: float,
    test_size: float,
    random_state: int,
) -> Tuple[DataFrame, DataFrame, DataFrame, DataFrame, DataFrame, DataFrame]:
    """
    Splits the data into train, validation, and test sets based on the specified ratios.

    Args:
        df (DataFrame): The input Pandas DataFrame.
        target_name (str): The name of target column.
        remainder_size (float): The ratio of validation and test data.
        test_size (float): The ratio of test data.
        random_state (int): The value of random seed.

    Returns:
        X_train (DataFrame): The train split DataFrame.
        X_val (DataFrame): The validation split DataFrame.
        X_test (DataFrame): The test split DataFrame.
        y_train (DataFrame): The train target split DataFrame.
        y_val (DataFrame): The validation target split DataFrame.
        y_test (DataFrame): The test target split DataFrame.

    """
    logger.info("Splitting data into train and test sets...")

    y = df[target_name].to_numpy().reshape(-1, 1)
    X = df.loc[:, ~df.columns.isin([target_name])]
    X_train, X_remain, y_train, y_remain = train_test_split(
        X, y, test_size=remainder_size, random_state=random_state
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_remain, y_remain, test_size=test_size, random_state=random_state
    )

    logger.info("Data split completed")
    return X_train, X_val, X_test, y_train, y_val, y_test


@task(retries=0, retry_delay_seconds=2)
def features_select(
    x_train: DataFrame, y_train: DataFrame, target_output_distribution: str
) -> List[str]:
    """
    Performs feature selection using Recursive Feature Elimination with Cross-Validation (RFECV).

    Args:
        x_train (DataFrame): The train split DataFrame.
        y_train: The train target split DataFrame.
        target_output_distribution (str): The name of expected output distribution

    Returns:
        selected_features (list): The list of selected features.
        pipeline (): .
        selected_features (): .
        scaler (): .
        transformer (): .

    """
    logger.info("Performing feature selection...")

    scaler = StandardScaler()
    transformer = QuantileTransformer(output_distribution=target_output_distribution)
    selector = RFECV(
        estimator=XGBRegressor(random_state=RANDOM_STATE),
        step=50,
        cv=5,
        scoring=METRIC,
        n_jobs=-1,
        verbose=1,
    )

    pipeline = Pipeline(
        steps=[
            ("scaler", scaler),
            ("transformer", transformer),
            ("selector", selector),
        ],
        verbose=1,
    )

    x_scaled = scaler.fit_transform(x_train[["Mileage", "Power"]])
    x_transformed = x_train.copy()
    x_transformed[["Mileage", "Power"]] = x_scaled

    y_transformed = transformer.fit_transform(y_train.reshape(-1, 1))

    pipeline.fit(x_transformed, y_transformed)

    selected_features = [
        feature
        for feature, selected in zip(
            x_train.columns, pipeline.named_steps["selector"].support_
        )
        if selected
    ]

    logger.info("Selected features:")
    for feature in selected_features:
        logger.info(feature)

    logger.info("Feature selection completed")
    return pipeline, selected_features, scaler, transformer


@task(retries=0, retry_delay_seconds=2)
def hyperparameters_gridsearch(
    x_train: DataFrame,
    y_train: DataFrame,
    x_val: DataFrame,
    y_val: DataFrame,
    regressor_grid: Dict,
    metric: str,
    selected_features: List[str],
    scaler: StandardScaler,
) -> Dict:
    """
    Performs hyperparameter tuning using GridSearchCV.

    Args:
        x_train (DataFrame): The training DataFrame.
        y_train (): .
        x_val (): .
        y_val (): .
        regressor_grid (): .
        metric (): .
        selected_features (): .
        scaler (): .

    Returns:
        results: ().

    """
    logger.info("Starting hyperparameter grid search...")

    regressors = {"XGBoost": XGBRegressor(n_jobs=-1)}
    parameters = regressor_grid
    results = {}

    for regressor_label, regressor in regressors.items():
        logger.info(f"Now training: {regressor_label}")

        # Apply scaling and transformation on validation data
        x_scaled_val = scaler.transform(x_val)
        y_transformed_val = y_val.reshape(-1, 1)

        steps = [("regressor", regressor)]
        pipeline = Pipeline(steps=steps)
        param_grid = parameters[regressor_label]

        gscv = GridSearchCV(
            regressor,
            param_grid,
            cv=5,
            n_jobs=-1,
            verbose=1,
            scoring=metric,
        )

        # Apply scaling and transformation on training data within GridSearchCV
        x_scaled_train = scaler.transform(x_train)
        y_transformed_train = y_train.reshape(-1, 1)

        selected_feature_indices = [
            x_train.columns.get_loc(feature) for feature in selected_features
        ]
        x_train_selected = x_scaled_train[:, selected_feature_indices]

        gscv.fit(x_train_selected, np.ravel(y_transformed_train))
        best_params = gscv.best_params_
        best_score = gscv.best_score_

        regressor.set_params(**best_params)

        x_val_selected = x_scaled_val[:, selected_feature_indices]
        y_pred = gscv.predict(x_val_selected)
        scoring = r2_score(y_transformed_val, y_pred)

        hyperparameters = {
            "Regressor": gscv,
            "Best Parameters": best_params,
            "Train": best_score,
            "Val": scoring,
        }

        logger.info(f"Validation results for {regressor_label}:")
        logger.info(f"  - Best Parameters: {best_params}")
        logger.info(f"  - Train Score: {best_score:.4f}")
        logger.info(f"  - Validation Score: {scoring:.4f}")

        results[regressor_label] = hyperparameters

    logger.info("Hyperparameter grid search completed")
    return results


@task(retries=0, retry_delay_seconds=2)
def model_train(
    x_train: DataFrame,
    y_train: DataFrame,
    selected_features: List[str],
    hyperparameters: Dict,
    scaler: StandardScaler,
) -> XGBRegressor:
    """
    Trains the final model using the selected features and hyperparameters.

    Args:
        x_train (DataFrame): The training DataFrame.
        y_train (): .
        selected_features (): .
        hyperparameters (): .
        scaler (): .

    Returns:
        model: The trained machine learning model.

    """
    logger.info("Training the model...")
    x_scaled_train = scaler.transform(x_train)
    y_transformed_train = y_train.reshape(-1, 1)

    selected_feature_indices = [
        x_train.columns.get_loc(feature) for feature in selected_features
    ]
    x_train_selected = x_scaled_train[:, selected_feature_indices]

    model = XGBRegressor(**hyperparameters["XGBoost"]["Best Parameters"])
    model.fit(x_train_selected, np.ravel(y_transformed_train))

    logger.info("Model training completed")
    return model


@task(retries=0, retry_delay_seconds=2)
def model_evaluate(
    model: XGBRegressor,
    x_test: DataFrame,
    y_test: DataFrame,
    selected_features: List[str],
    scaler: StandardScaler,
) -> Tuple[float, float]:
    """
    Evaluates the model on the test set.

    Args:
        model: The trained machine learning model.
        x_test (DataFrame): The test DataFrame.
        y_test (): .
        selected_features (): .
        scaler (): .

    Returns:
        mse (): The evaluation results.
        r2 (): The evaluation results.

    """

    logger.info("Evaluating the model...")

    # Apply the same transformations to the test data using the pipeline
    x_scaled_test = scaler.transform(x_test)

    selected_feature_indices = [
        x_test.columns.get_loc(feature) for feature in selected_features
    ]
    x_test_selected = x_scaled_test[:, selected_feature_indices]

    # Predict using the trained model
    y_pred = model.predict(x_test_selected)

    # Save CSV files
    pd.DataFrame(x_test_selected, columns=selected_features).to_csv(
        X_TEST_PATH, index=False
    )
    pd.DataFrame(y_test, columns=["Price"]).to_csv(Y_TEST_PATH, index=False)
    pd.DataFrame(y_pred, columns=["predictions"]).to_csv(Y_PRED_PATH, index=False)

    # Concatenate the CSV files
    x_test_selected_df = pd.DataFrame(x_test_selected, columns=selected_features)
    concatenated_df = pd.concat(
        [
            x_test_selected_df,
            pd.DataFrame(y_test, columns=["Price"]),
            pd.DataFrame(y_pred, columns=["predictions"]),
        ],
        axis=1,
    )
    concatenated_df.to_csv(CURRENT_PATH, index=False)

    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    logger.info(f"MSE: {mse}")
    logger.info(f"R^2: {r2}")
    logger.info("Model evaluation completed")

    return mse, r2


@task(retries=0, retry_delay_seconds=2)
def model_save(model: XGBRegressor, model_path: str) -> None:
    """
    Saves the trained model.

    Args:
        model: The trained machine learning model.
        model_path (str): The path to save the model.

    Returns:
        None

    """
    model.save_model(model_path)


@flow
def otomoto_training_flow():
    TRACKING_SERVER_HOST = "34.77.180.77"
    mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:5000")
    mlflow.set_experiment("otomoto-used-car-price-prediction")

    with mlflow.start_run():
        mlflow.autolog()

        # Download offers.csv file
        files_download(
            bucket_name=BUCKET_NAME,
            source_path=os.path.join(SOURCE_PATH, FILE_NAME),
            destination_path=os.path.join(DESTINATION_PATH, FILE_NAME),
        )

        # Create Spark session
        session = session_create(
            spark_session_scope=SPARK_SESSION_SCOPE,
            spark_session_name=SPARK_SESSION_NAME,
        )

        # Read data
        data_input = data_read(
            spark_session=session,
            header=HEADER,
            infer_schema=INFER_SCHEMA,
            file_path=FILE_PATH,
        )

        # Filter data
        data_filtered = data_filter(df=data_input)

        # Preprocess data
        data_preprocessed = data_preprocess(df=data_filtered)

        data_engineered = features_engineer(
            df=data_preprocessed,
            distinct_columns=DISTINCT_COLUMNS,
            columns_to_drop=COLUMNS_TO_DROP,
            brand_columns=BRAND_COLUMNS,
            fuel_columns=FUEL_COLUMNS,
            body_columns=BODY_COLUMNS,
            door_columns=DOOR_COLUMNS,
            file_preprocessed_path=FILE_PREPROCESSED_PATH,
        )

        # train_data, test_data = data_split(data_engineered, TEST_SIZE, RANDOM_STATE)

        X_train, X_val, X_test, y_train, y_val, y_test = data_split(
            df=data_engineered,
            target_name=TARGET_NAME,
            remainder_size=REMAINDER_SIZE,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
        )

        # Perform feature selection
        pipeline, selected_features, scaler, transformer = features_select(
            x_train=X_train,
            y_train=y_train,
            target_output_distribution=TARGET_OUTPUT_DISTRIBUTION,
        )

        # Perform hyperparameter tuning
        hyperparameters = hyperparameters_gridsearch(
            x_train=X_train,
            y_train=y_train,
            x_val=X_val,
            y_val=y_val,
            regressor_grid=REGRESSOR_GRID,
            metric=METRIC,
            selected_features=selected_features,
            scaler=scaler,
        )

        # Train the final model
        model_trained = model_train(
            x_train=X_train,
            y_train=y_train,
            selected_features=selected_features,
            hyperparameters=hyperparameters,
            scaler=scaler,
        )

        # Evaluate the model
        model_evaluate(
            model=model_trained,
            x_test=X_test,
            y_test=y_test,
            selected_features=selected_features,
            scaler=scaler,
        )

        # Save the model
        model_save(
            model=model_trained,
            model_path=MODEL_PATH,
        )

        autolog_run = mlflow.last_active_run()


if __name__ == "__main__":
    otomoto_training_flow()
