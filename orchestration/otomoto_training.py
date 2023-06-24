import os

os.environ[
    "GOOGLE_APPLICATION_CREDENTIALS"
] = "/home/konradballegro/.ssh/ny-rides-konrad-b5129a6f2e66.json"
import numpy as np
import pandas as pd
import pyspark
from google.cloud import storage
from prefect import flow, task
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_replace, when
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from xgboost import XGBRegressor


# copy offercs.csv from GCP bucket to VM
@task(retries=0, retry_delay_seconds=2)
def download_offers(bucket_name, source_path, destination_path):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_path)
    blob.download_to_filename(destination_path)


@task(retries=0, retry_delay_seconds=2)
def read_data():
    spark = (
        SparkSession.builder.master("local[*]")
        .appName("otomoto_preprocessing")
        .getOrCreate()
    )

    df = (
        spark.read.option("header", "true")
        .option("inferSchema", "True")
        .csv("/home/konradballegro/orchestration/data/inputs/offers.csv")
    )

    df = df.filter(
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
    ).select(
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

    distinct_offers = (
        df.select("Offer from").distinct().rdd.flatMap(lambda x: x).collect()
    )
    distinct_conditions = (
        df.select("Condition").distinct().rdd.flatMap(lambda x: x).collect()
    )
    distinct_brands = (
        df.select("Vehicle brand").distinct().rdd.flatMap(lambda x: x).collect()
    )
    distinct_years = (
        df.select("Year of production").distinct().rdd.flatMap(lambda x: x).collect()
    )
    distinct_fuel = df.select("Fuel type").distinct().rdd.flatMap(lambda x: x).collect()
    distinct_gearbox = (
        df.select("Gearbox").distinct().rdd.flatMap(lambda x: x).collect()
    )
    distinct_body = df.select("Body type").distinct().rdd.flatMap(lambda x: x).collect()
    distinct_doors = (
        df.select("Number of doors").distinct().rdd.flatMap(lambda x: x).collect()
    )

    for offer in distinct_offers:
        column_name = "Offer_type_" + offer.replace(" ", "_")
        df = df.withColumn(column_name, when(df["Offer from"] == offer, 1).otherwise(0))

    for condition in distinct_conditions:
        column_name = "Condition_" + condition.replace(" ", "_")
        df = df.withColumn(
            column_name, when(df["Condition"] == condition, 1).otherwise(0)
        )

    for brand in distinct_brands:
        column_name = "Vehicle_brand_" + brand.replace(" ", "_")
        df = df.withColumn(
            column_name, when(df["Vehicle brand"] == brand, 1).otherwise(0)
        )

    for year in distinct_years:
        column_name = "Year_of_production_" + str(year)
        df = df.withColumn(
            column_name, when(df["Year of production"] == year, 1).otherwise(0)
        )

    for fuel in distinct_fuel:
        column_name = "Fuel_type_" + fuel.replace(" ", "_")
        df = df.withColumn(column_name, when(df["Fuel type"] == fuel, 1).otherwise(0))

    for gearbox in distinct_gearbox:
        column_name = "Gearbox_" + gearbox.replace(" ", "_")
        df = df.withColumn(column_name, when(df["Gearbox"] == gearbox, 1).otherwise(0))

    for body in distinct_body:
        column_name = "Body_type_" + body.replace(" ", "_")
        df = df.withColumn(column_name, when(df["Body type"] == body, 1).otherwise(0))

    for doors in distinct_doors:
        column_name = "Number_of_doors_" + str(doors)
        df = df.withColumn(
            column_name, when(df["Number of doors"] == doors, 1).otherwise(0)
        )

    # Assuming your DataFrame is called df
    columns_to_drop = [
        "Offer from",
        "Condition",
        "Vehicle brand",
        "Vehicle model",
        "Year of production",
        "Fuel type",
        "Gearbox",
        "Body type",
        "Number of doors",
        "URL path",
        "ID",
        "Epoch",
    ]

    # Drop the specified columns
    df = df.drop(*columns_to_drop)

    df = df.filter(df["Price"].isNotNull())

    df_pandas = df.toPandas()

    # print(df_pandas.head())

    df_pandas.to_csv("/home/konradballegro/notebooks/outputs/data/offers_filtered.csv")

    # preprocess data with spark

    # Convert the pandas Series to a NumPy array and reshape it
    y = df_pandas["Price"]
    y_array = y.to_numpy().reshape(-1, 1)

    # Quantile transformation
    transformer = QuantileTransformer(output_distribution="normal")
    y_transformed = transformer.fit_transform(y_array)

    y = y_transformed

    print("y.max(): ", y.max())
    print("y.min(): ", y.min())

    X = df_pandas.loc[:, ~df_pandas.columns.isin(["Price"])]

    # Split data into train and remaining data
    X_train, X_remain, y_train, y_remain = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Split remaining data into validation and test sets
    X_val, X_test, y_val, y_test = train_test_split(
        X_remain, y_remain, test_size=0.5, random_state=42
    )

    print("Shape of X_train:", X_train.shape)
    print("Shape of X_val:", X_val.shape)
    print("Shape of X_test:", X_test.shape)
    print("Shape of y_train:", y_train.shape)
    print("Shape of y_val:", y_val.shape)
    print("Shape of y_test:", y_test.shape)

    # Assuming df is your DataFrame containing the data
    subset = X_train[["Mileage", "Power"]]

    scaler = StandardScaler()
    scaler.fit(subset)
    subset_scaled = scaler.transform(subset)

    X_train["Mileage"] = subset_scaled[:, 0]
    X_train["Power"] = subset_scaled[:, 1]

    corr_matrix = X_train.corr(method="spearman").abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]
    X_train = X_train.drop(to_drop, axis=1)
    X_val = X_val.drop(to_drop, axis=1)
    X_test = X_test.drop(to_drop, axis=1)

    selected_features = [
        "Mileage",
        "Power",
        "Offer_type_Osoby_prywatnej",
        #     'Vehicle_brand_Infiniti',
        #     'Vehicle_brand_Lexus',
        #     'Vehicle_brand_Jaguar',
        #     'Vehicle_brand_Maserati',
        #     'Vehicle_brand_Jeep',
        #     'Vehicle_brand_Lancia',
        #     'Vehicle_brand_Kia',
        #     'Vehicle_brand_Hyundai',
        #     'Vehicle_brand_Honda',
        #     'Vehicle_brand_Lamborghini',
        #     'Vehicle_brand_Ligier',
        #     'Vehicle_brand_Isuzu',
        #     'Vehicle_brand_Land_Rover',
        "Vehicle_brand_Mercedes-Benz",
        #     'Vehicle_brand_McLaren',
        "Gearbox_Manualna",
        "Number_of_doors_2",
        "Number_of_doors_5",
        #     'Number_of_doors_6',
        "Fuel_type_Benzyna",
        "Fuel_type_Benzyna+LPG",
        "Fuel_type_Hybryda",
        #     'Fuel_type_Elektryczny',
        "Body_type_SUV",
        "Body_type_Minivan",
        "Body_type_Coupe",
    ]

    X_train = X_train[selected_features]
    X_val = X_val[selected_features]
    X_test = X_test[selected_features]

    model = XGBRegressor()
    model.load_model("/home/konradballegro/orchestration/models/xgb.model")

    # Use the loaded model for predictions
    y_pred = model.predict(X_test)

    # Evaluate the model using mean squared error
    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error:", mse)

    # Evaluate the model using mean squared error
    r2 = r2_score(y_test, y_pred)
    print("R2:", r2)


# scale two columns

# add feature engineering, including interactions between continues variables

# normalize skewed target

# split data into train, val, test

# train and save xgb model using MlFlow and preselected hyperparameters

# define main flow and tracking server


@flow
def otomoto_training_flow():
    bucket_name = "mlops-zoomcamp"
    source_path = "data/training/outputs"
    destination_path = "/home/konradballegro/orchestration/data/inputs"
    file_name = "offers.csv"

    download_offers(
        bucket_name=bucket_name,
        source_path=f"{source_path}/{file_name}",
        destination_path=f"{destination_path}/{file_name}",
    )

    read_data()


if __name__ == "__main__":
    print("hello world")
    otomoto_training_flow()
