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
    REGRESSOR_PARAMETERS = config["REGRESSOR_PARAMETERS"]
    METRIC = config["METRIC"]


# Copy offercs.csv from GCP bucket to VM
@task(retries=0, retry_delay_seconds=2)
def offers_download(bucket_name, source_path, destination_path):
    from google.cloud import storage

    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(source_path)
    blob.download_to_filename(destination_path)


@task(retries=0, retry_delay_seconds=2)
def spark_session_create(spark_session_scope, spark_session_name):
    session = (
        SparkSession.builder.master(spark_session_scope)
        .appName(spark_session_name)
        .getOrCreate()
    )

    return session


@task(retries=0, retry_delay_seconds=2)
def data_read(spark_session, header, infer_schema, file_name):
    return (
        spark_session.read.option("header", header)
        .option("inferSchema", infer_schema)
        .csv(file_name)
    )


@task(retries=0, retry_delay_seconds=2)
def data_filter(df):
    df_filtered = df.filter(
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

    return df_filtered


@task(retries=0, retry_delay_seconds=2)
def data_clean(df):
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

    return df_cleaned


@task(retries=0, retry_delay_seconds=2)
def features_engineer(
    df,
    distinct_columns,
    columns_to_drop,
    brand_columns,
    fuel_columns,
    body_columns,
    door_columns,
    offers_preprocessed_path,
):
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

    # Production Year Category
    # year_columns = YEAR_COLUMNS
    # df_pd["production_year_category"] = pd.cut(
    #     df_pd[year_columns].sum(axis=1),
    #     bins=[1950, 1990, 2000, 2010, 2025],
    #     labels=["Older", "Mid-range", "Newer", "Future"],
    # )

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

    return df_pd


@task(retries=0, retry_delay_seconds=2)
def target_transform(df, target_name, target_output_distribution):
    y = df[target_name].to_numpy().reshape(-1, 1)
    transformer = QuantileTransformer(output_distribution=target_output_distribution)
    y_transformed = transformer.fit_transform(y)
    y = y_transformed

    return df


@task(retries=0, retry_delay_seconds=2)
def data_split(df, target_name, remainder_size, test_size, random_state):
    y = df[target_name].to_numpy().reshape(-1, 1)
    print("y.min()", y.min())
    print("y.max()", y.max())
    X = df.loc[:, ~df.columns.isin([target_name])]
    X_train, X_remain, y_train, y_remain = train_test_split(
        X, y, test_size=remainder_size, random_state=random_state
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_remain, y_remain, test_size=test_size, random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


@task(retries=0, retry_delay_seconds=2)
def data_scale(x_train):
    subset = x_train[["Mileage", "Power"]]
    scaler = StandardScaler()
    subset_scaled = scaler.fit_transform(subset)
    x_train["Mileage"] = subset_scaled[:, 0]
    x_train["Power"] = subset_scaled[:, 1]

    return x_train


@task(retries=0, retry_delay_seconds=2)
def correlated_drop(x_train, x_val, x_test):
    corr_matrix = x_train.corr(method="spearman").abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]
    x_train = x_train.drop(to_drop, axis=1)
    x_val = x_val.drop(to_drop, axis=1)
    x_test = x_test.drop(to_drop, axis=1)

    return x_train, x_val, x_test


@task(retries=0, retry_delay_seconds=2)
def features_select(x_train, x_val, x_test, selected_features):
    x_train = x_train[selected_features]
    x_val = x_val[selected_features]
    x_test = x_test[selected_features]

    return x_train, x_val, x_test


@task(retries=0, retry_delay_seconds=2)
def hyperparameters_gridsearch(
    x_train, y_train, x_val, y_val, regressor_parameters, metric
):
    regressors = {}
    regressors.update({"XGBoost": XGBRegressor(n_jobs=-1)})

    parameters = {}
    parameters.update(regressor_parameters)

    results = {}
    for regressor_label, regressor in regressors.items():
        print(f"Now training: {regressor_label}")
        steps = [("regressor", regressor)]
        pipeline = Pipeline(steps=steps, verbose=1)
        param_grid = parameters[regressor_label]

        gscv = GridSearchCV(
            regressor,
            param_grid,
            cv=5,
            n_jobs=-1,
            verbose=1,
            # scoring="neg_root_mean_squared_error",
            scoring=metric,
        )

        gscv.fit(x_train, np.ravel(y_train))
        best_params = gscv.best_params_
        best_score = gscv.best_score_

        regressor.set_params(**best_params)

        y_pred = gscv.predict(x_val)
        scoring = mean_squared_error(y_val, y_pred)

        hyperparameters = {
            "Regressor": gscv,
            "Best Parameters": best_params,
            "Train": best_score,
            "Val": scoring,
        }

        print("Val:", "{:.4f}".format(hyperparameters["Val"]))

        results.update({regressor_label: hyperparameters})

        return results


@task(retries=0, retry_delay_seconds=2)
def model_train(x_train, y_train, x_test, y_test, hyperparameters):
    model = XGBRegressor(**hyperparameters["XGBoost"]["Best Parameters"])

    # Fit the regressor to the training data
    model.fit(x_train, y_train)

    # loading saved model
    # model.load_model(MODEL_PATH)
    y_pred = model.predict(x_test)

    # scoring
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Mean Squared Error:", mse)
    print("R2:", r2)


@flow
def otomoto_training_flow():
    offers_download(
        bucket_name=BUCKET_NAME,
        source_path=os.path.join(SOURCE_PATH, FILE_NAME),
        destination_path=os.path.join(DESTINATION_PATH, FILE_NAME),
    )

    session = spark_session_create(
        spark_session_scope=SPARK_SESSION_SCOPE, spark_session_name=SPARK_SESSION_NAME
    )

    data_input = data_read(
        spark_session=session,
        header=HEADER,
        infer_schema=INFER_SCHEMA,
        file_name=OFFERS_PATH,
    )

    data_filtered = data_filter(df=data_input)

    data_cleaned = data_clean(df=data_filtered)

    data_engineered = features_engineer(
        df=data_cleaned,
        distinct_columns=DISTINCT_COLUMNS,
        columns_to_drop=COLUMNS_TO_DROP,
        brand_columns=BRAND_COLUMNS,
        fuel_columns=FUEL_COLUMNS,
        body_columns=BODY_COLUMNS,
        door_columns=DOOR_COLUMNS,
        offers_preprocessed_path=OFFERS_PREPROCESSED_PATH,
    )

    data_transformed = target_transform(
        df=data_engineered,
        target_name=TARGET_NAME,
        target_output_distribution=TARGET_OUTPUT_DISTRIBUTION,
    )

    X_train, X_val, X_test, y_train, y_val, y_test = data_split(
        df=data_transformed,
        target_name=TARGET_NAME,
        remainder_size=REMAINDER_SIZE,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    X_train = data_scale(x_train=X_train)

    X_train, X_val, X_test = correlated_drop(
        x_train=X_train, x_val=X_val, x_test=X_test
    )

    X_train, X_val, X_test = features_select(
        x_train=X_train, x_val=X_val, x_test=X_test, selected_features=SELECTED_FEATURES
    )

    hyperparameters = hyperparameters_gridsearch(
        x_train=X_train,
        x_val=X_val,
        y_train=y_train,
        y_val=y_val,
        regressor_parameters=REGRESSOR_PARAMETERS,
        metric=METRIC,
    )

    model_train(
        x_train=X_train,
        x_test=X_test,
        y_train=y_train,
        y_test=y_test,
        hyperparameters=hyperparameters,
    )


if __name__ == "__main__":
    print("hello world")
    otomoto_training_flow()
