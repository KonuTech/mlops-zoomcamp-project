import json
import logging
import os
from functools import reduce

import numpy as np
import pandas as pd
import pyspark
from google.cloud import storage
from prefect import flow, task
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_replace, when
from sklearn.feature_selection import RFE, RFECV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from xgboost import XGBRegressor

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
OFFERS_PATH = config["OFFERS_PATH"]
OFFERS_PREPROCESSED_PATH = config["OFFERS_PREPROCESSED_PATH"]
TARGET_NAME = config["TARGET_NAME"]
TARGET_OUTPUT_DISTRIBUTION = config["TARGET_OUTPUT_DISTRIBUTION"]
SELECTED_FEATURES = config["SELECTED_FEATURES"]
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
def offers_download(bucket_name, source_path, destination_path):
    logger.info(f"Downloading file from bucket '{bucket_name}' to '{destination_path}'")
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_path)
    blob.download_to_filename(destination_path)
    logger.info("File downloaded successfully")


@task(retries=0, retry_delay_seconds=2)
def session_create(spark_session_scope, spark_session_name):
    logger.info("Creating Spark session...")
    session = (
        SparkSession.builder.master(spark_session_scope)
        .appName(spark_session_name)
        .getOrCreate()
    )
    logger.info("Spark session created")
    return session


@task(retries=0, retry_delay_seconds=2)
def data_read(spark_session, header, infer_schema, file_name):
    logger.info(f"Reading data from file: '{file_name}'")
    return (
        spark_session.read.option("header", header)
        .option("inferSchema", infer_schema)
        .csv(file_name)
    )


@task(retries=0, retry_delay_seconds=2)
def data_filter(df):
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
def data_preprocessing(df):
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
def features_engineering(
    df,
    distinct_columns,
    columns_to_drop,
    brand_columns,
    fuel_columns,
    body_columns,
    door_columns,
    offers_preprocessed_path,
):
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

    df_pd.to_csv(offers_preprocessed_path, index=False)
    logger.info("Feature engineering completed")
    return df_pd


@task(retries=0, retry_delay_seconds=2)
def data_split(df, target_name, remainder_size, test_size, random_state):
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
def feature_selection(x_train, y_train, target_output_distribution):
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
    x_train, y_train, x_val, y_val, regressor_grid, metric, selected_features, scaler
):
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

        gscv.fit(x_scaled_train, np.ravel(y_transformed_train))
        best_params = gscv.best_params_
        best_score = gscv.best_score_

        regressor.set_params(**best_params)

        y_pred = gscv.predict(x_scaled_val)
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
def model_training(x_train, y_train, hyperparameters, scaler):
    logger.info("Training the model...")
    x_scaled_train = scaler.transform(x_train)
    y_transformed_train = y_train.reshape(-1, 1)

    model = XGBRegressor(**hyperparameters["XGBoost"]["Best Parameters"])
    model.fit(x_scaled_train, np.ravel(y_transformed_train))

    logger.info("Model training completed")
    return model


@task(retries=0, retry_delay_seconds=2)
def model_evaluation(model, x_test, y_test, scaler):
    logger.info("Evaluating the model...")

    # Apply the same transformations to the test data using the pipeline
    x_scaled_test = scaler.transform(x_test)

    # Reshape y_test to match the expected format
    y_transformed_test = y_test.reshape(-1, 1)

    # Predict using the trained model in the pipeline
    y_pred = model.predict(x_scaled_test)

    # Inverse transform the predicted values using the target scaler
    # y_pred = pipeline.named_steps["transformer"].(y_pred_transformed)
    pd.DataFrame(x_scaled_test).to_csv(
        "/home/konradballegro/notebooks/outputs/data/x_scaled_test.csv",
        index=False,
    )
    pd.DataFrame(y_transformed_test).to_csv(
        "/home/konradballegro/notebooks/outputs/data/y_transformed_test.csv", index=False
    )
    pd.DataFrame(y_pred).to_csv(
        "/home/konradballegro/notebooks/outputs/data/y_pred.csv", index=False
    )
    # Calculate evaluation metrics
    mse = mean_squared_error(y_transformed_test, y_pred)
    r2 = r2_score(y_transformed_test, y_pred)

    logger.info(f"MSE: {mse}")
    logger.info(f"R^2: {r2}")
    logger.info("Model evaluation completed")

    # Return any evaluation results as needed
    return mse, r2


@task(retries=0, retry_delay_seconds=2)
def model_saving(model, model_path):
    model.save_model(model_path)


@flow
def otomoto_training_flow():
    offers_download(
        bucket_name=BUCKET_NAME,
        source_path=os.path.join(SOURCE_PATH, FILE_NAME),
        destination_path=os.path.join(DESTINATION_PATH, FILE_NAME),
    )

    session = session_create(
        spark_session_scope=SPARK_SESSION_SCOPE, spark_session_name=SPARK_SESSION_NAME
    )

    data_input = data_read(
        spark_session=session,
        header=HEADER,
        infer_schema=INFER_SCHEMA,
        file_name=OFFERS_PATH,
    )

    data_filtered = data_filter(df=data_input)

    data_preprocessed = data_preprocessing(df=data_filtered)

    data_engineered = features_engineering(
        df=data_preprocessed,
        distinct_columns=DISTINCT_COLUMNS,
        columns_to_drop=COLUMNS_TO_DROP,
        brand_columns=BRAND_COLUMNS,
        fuel_columns=FUEL_COLUMNS,
        body_columns=BODY_COLUMNS,
        door_columns=DOOR_COLUMNS,
        offers_preprocessed_path=OFFERS_PREPROCESSED_PATH,
    )

    # train_data, test_data = data_split(data_engineered, TEST_SIZE, RANDOM_STATE)

    X_train, X_val, X_test, y_train, y_val, y_test = data_split(
        df=data_engineered,
        target_name=TARGET_NAME,
        remainder_size=REMAINDER_SIZE,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    pipeline, selected_features, scaler, transformer = feature_selection(
        x_train=X_train,
        y_train=y_train,
        target_output_distribution=TARGET_OUTPUT_DISTRIBUTION,
    )

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

    # Train the model
    model_trained = model_training(
        x_train=X_train, y_train=y_train, hyperparameters=hyperparameters, scaler=scaler
    )

    # Evaluate the model
    model_evaluation(
        model=model_trained,
        x_test=X_test,
        y_test=y_test,
        scaler=scaler,
    )

    # Save the model
    model_saving(
        model=model_trained,
        model_path=MODEL_PATH,
    )


if __name__ == "__main__":
    otomoto_training_flow()
