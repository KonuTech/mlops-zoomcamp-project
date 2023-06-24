import json
import os

import numpy as np
import pandas as pd
import pyspark
from prefect import flow, task
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_replace, when
from sklearn.feature_selection import RFE, RFECV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, ParameterGrid, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from xgboost import XGBRegressor

os.environ[
    "GOOGLE_APPLICATION_CREDENTIALS"
] = "/home/konradballegro/.ssh/ny-rides-konrad-b5129a6f2e66.json"

with open("/home/konradballegro/orchestration/config/config.json") as json_file:
    config = json.load(json_file)
    BUCKET_NAME = config["BUCKET_NAME"]
    SOURCE_PATH = config["SOURCE_PATH"]
    DESTINATION_PATH = config["DESTINATION_PATH"]
    FILE_NAME = config["FILE_NAME"]
    SPARK_SESSION_SCOPE = config["SPARK_SESSION_SCOPE"]
    SPARK_SESSION_NAME = config["SPARK_SESSION_NAME"]
    MODEL_PATH = config["MODEL_PATH"]
    OFFERS = config["OFFERS"]
    OFFERS_FILTERED = config["OFFERS_FILTERED"]
    SELECTED_FEATURES = config["SELECTED_FEATURES"]
    DISTINCT_COLUMNS = config["DISTINCT_COLUMNS"]
    COLUMNS_TO_DROP = config["COLUMNS_TO_DROP"]
    DOOR_COLUMNS = config["DOOR_COLUMNS"]
    BODY_COLUMNS = config["BODY_COLUMNS"]
    FUEL_COLUMNS = config["FUEL_COLUMNS"]
    BRAND_COLUMNS = config["BRAND_COLUMNS"]
    YEAR_COLUMNS = config["YEAR_COLUMNS"]


# Copy offercs.csv from GCP bucket to VM
@task(retries=0, retry_delay_seconds=2)
def download_offers(bucket_name, source_path, destination_path):
    from google.cloud import storage

    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(source_path)
    blob.download_to_filename(destination_path)


@task(retries=0, retry_delay_seconds=2)
def read_data():
    spark = (
        SparkSession.builder.master(SPARK_SESSION_SCOPE)
        .appName(SPARK_SESSION_NAME)
        .getOrCreate()
    )

    return spark.read.option("header", "true").option("inferSchema", "True").csv(OFFERS)


def filter_data(df):
    df_filtered = df.select(
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
    ).filter(
        (df["Currency"] == "PLN")
        & (df["Country of origin"] == "Polska")
        & (df["Accident-free"].isNotNull())
        & (df["Price"].isNotNull())
        & (df["Offer from"].isNotNull())
        & (df["Condition"].isNotNull())
        & (df["Condition"] == "Używane")
        & (df["Vehicle brand"].isNotNull())
        & (df["Year of production"].isNotNull())
        & (df["Mileage"].isNotNull())
        & (df["Fuel type"].isNotNull())
        & (df["Power"].isNotNull())
        & (df["Gearbox"].isNotNull())
        & (df["Body type"].isNotNull())
        & (df["Number of doors"].isNotNull())
    )

    distinct_columns = DISTINCT_COLUMNS

    for column in distinct_columns:
        distinct_values = (
            df_filtered.select(column).distinct().rdd.flatMap(lambda x: x).collect()
        )
        for value in distinct_values:
            column_name = f"{column.replace(' ', '_')}_{value.replace(' ', '_')}"
            df_filtered = df_filtered.withColumn(
                column_name, when(df_filtered[column] == value, 1).otherwise(0)
            )

    columns_to_drop = COLUMNS_TO_DROP

    df_filtered = df_filtered.drop(*columns_to_drop).filter(
        df_filtered["Price"].isNotNull()
    )

    df_pandas = df_filtered.toPandas()

    # Perform feature engineering
    # Price per Mileage
    df_pandas["price_per_mileage"] = df_pandas["Price"] / df_pandas["Mileage"]

    # Power-to-Price Ratio
    df_pandas["power_to_price_ratio"] = df_pandas["Power"] / df_pandas["Price"]

    # Number of Brands
    brand_columns = BRAND_COLUMNS
    df_pandas["brand_ratio"] = df_pandas[brand_columns].sum(axis=1) / len(brand_columns)

    # Production Year Category
    # year_columns = YEAR_COLUMNS
    # df_pandas["production_year_category"] = pd.cut(
    #     df_pandas[year_columns].sum(axis=1),
    #     bins=[1950, 1990, 2000, 2010, 2025],
    #     labels=["Older", "Mid-range", "Newer", "Future"],
    # )

    # Fuel Type Count
    fuel_columns = FUEL_COLUMNS
    df_pandas["fuel_type_ratio"] = df_pandas[fuel_columns].sum(axis=1) / len(
        fuel_columns
    )

    # Body Type Count
    body_columns = BODY_COLUMNS
    df_pandas["body_type_ratio"] = df_pandas[body_columns].sum(axis=1) / len(
        body_columns
    )

    # Door Count
    door_columns = DOOR_COLUMNS
    df_pandas["door_number_ratio"] = df_pandas[door_columns].sum(axis=1) / len(
        door_columns
    )

    df_pandas.to_csv(OFFERS_FILTERED, index=False)

    y = df_pandas["Price"].to_numpy().reshape(-1, 1)
    transformer = QuantileTransformer(output_distribution="normal")
    y_transformed = transformer.fit_transform(y)
    y = y_transformed

    X = df_pandas.loc[:, ~df_pandas.columns.isin(["Price"])]
    X_train, X_remain, y_train, y_remain = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_remain, y_remain, test_size=0.5, random_state=42
    )

    subset = X_train[["Mileage", "Power"]]
    scaler = StandardScaler()
    subset_scaled = scaler.fit_transform(subset)
    X_train["Mileage"] = subset_scaled[:, 0]
    X_train["Power"] = subset_scaled[:, 1]

    corr_matrix = X_train.corr(method="spearman").abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]
    X_train = X_train.drop(to_drop, axis=1)
    X_val = X_val.drop(to_drop, axis=1)
    X_test = X_test.drop(to_drop, axis=1)

    selected_features = SELECTED_FEATURES

    X_train = X_train[selected_features]
    X_val = X_val[selected_features]
    X_test = X_test[selected_features]

    regressors = {}
    regressors.update({"XGBoost": XGBRegressor(n_jobs=-1)})

    parameters = {}
    parameters.update(
        {
            "XGBoost": {
                "colsample_bytree": [0.5],
                "learning_rate": [0.1],
                "max_depth": [8],
                "n_estimators": [200],
                "subsample": [1.0],
            }
        }
    )

    results = {}
    for regressor_label, regressor in regressors.items():
        print("\n" + f"Teraz trenuje: {regressor_label}")

        steps = [("regressor", regressor)]
        pipeline = Pipeline(steps=steps, verbose=1)

        param_grid = parameters[regressor_label]  # Access the parameter grid correctly

        gscv = GridSearchCV(
            regressor,
            param_grid,
            cv=5,
            n_jobs=-1,
            verbose=1,
            # scoring="neg_root_mean_squared_error",
            scoring="r2",
        )

        gscv.fit(X_train, np.ravel(y_train))
        best_params = gscv.best_params_
        best_score = gscv.best_score_

        regressor.set_params(**best_params)

        y_pred = gscv.predict(X_val)
        scoring = mean_squared_error(y_val, y_pred)

        result = {
            "Regressor": gscv,
            "Best Parameters": best_params,
            "Train": best_score,
            "Val": scoring,
        }

        print("Val:", "{:.4f}".format(result["Val"]))

        results.update({regressor_label: result})

    model = XGBRegressor(**results["XGBoost"]["Best Parameters"])

    # Fit the regressor to the training data
    model.fit(X_train, y_train)

    # loading saved model
    model.load_model(MODEL_PATH)
    y_pred = model.predict(X_test)

    # scoring
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Mean Squared Error:", mse)
    print("R2:", r2)


@flow
def otomoto_training_flow():
    download_offers(
        bucket_name=BUCKET_NAME,
        source_path=os.path.join(SOURCE_PATH, FILE_NAME),
        destination_path=os.path.join(DESTINATION_PATH, FILE_NAME),
    )

    data_input = read_data()
    filter_data(df=data_input)


if __name__ == "__main__":
    print("hello world")
    otomoto_training_flow()
