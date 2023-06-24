import os

import numpy as np
import pandas as pd
from prefect import flow, task
from sklearn.feature_selection import RFE, RFECV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, ParameterGrid, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from xgboost import XGBRegressor

os.environ[
    "GOOGLE_APPLICATION_CREDENTIALS"
] = "/home/konradballegro/.ssh/ny-rides-konrad-b5129a6f2e66.json"


# Copy offercs.csv from GCP bucket to VM
@task(retries=0, retry_delay_seconds=2)
def download_offers(bucket_name, source_path, destination_path):
    from google.cloud import storage

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_path)
    blob.download_to_filename(destination_path)


@task(retries=0, retry_delay_seconds=2)
def read_data():
    import pyspark
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import col, regexp_replace, when

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

    selected_features = [
        "Mileage",
        "Power",
        "Offer_from_Osoby_prywatnej",
        "Vehicle_brand_Mercedes-Benz",
        "Gearbox_Manualna",
        "Number_of_doors_2",
        "Number_of_doors_5",
        "Fuel_type_Benzyna",
        "Fuel_type_Benzyna+LPG",
        "Fuel_type_Hybryda",
        "Body_type_SUV",
        "Body_type_Minivan",
        "Body_type_Coupe",
        # "price_per_mileage",
        # "power_to_price_ratio",
        # "brand_ratio",
        # "production_year_category",
        # "fuel_type_ratio",
        # "body_type_ratio",
        # "door_number_ratio",
    ]

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

    distinct_columns = [
        "Offer from",
        "Condition",
        "Vehicle brand",
        "Year of production",
        "Fuel type",
        "Gearbox",
        "Body type",
        "Number of doors",
    ]

    for column in distinct_columns:
        distinct_values = (
            df_filtered.select(column).distinct().rdd.flatMap(lambda x: x).collect()
        )
        for value in distinct_values:
            column_name = f"{column.replace(' ', '_')}_{value.replace(' ', '_')}"
            df_filtered = df_filtered.withColumn(
                column_name, when(df_filtered[column] == value, 1).otherwise(0)
            )

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
    brand_columns = [
        "Vehicle_brand_Infiniti",
        "Vehicle_brand_Lexus",
        "Vehicle_brand_Jaguar",
        "Vehicle_brand_Maserati",
        "Vehicle_brand_Jeep",
        "Vehicle_brand_Lancia",
        "Vehicle_brand_Kia",
        "Vehicle_brand_Hyundai",
        "Vehicle_brand_Honda",
        "Vehicle_brand_Lamborghini",
        "Vehicle_brand_Ligier",
        "Vehicle_brand_Isuzu",
        "Vehicle_brand_Land_Rover",
        "Vehicle_brand_Mercedes-Benz",
        "Vehicle_brand_McLaren",
        "Vehicle_brand_Lada",
        "Vehicle_brand_Iveco",
        "Vehicle_brand_Mazda",
        "Vehicle_brand_Maybach",
        "Vehicle_brand_Peugeot",
        "Vehicle_brand_Mitsubishi",
        "Vehicle_brand_Microcar",
        "Vehicle_brand_MINI",
        "Vehicle_brand_Opel",
        "Vehicle_brand_Polonez",
        "Vehicle_brand_Porsche",
        "Vehicle_brand_Nissan",
        "Vehicle_brand_Tarpan",
        "Vehicle_brand_Rover",
        "Vehicle_brand_Rolls-Royce",
        "Vehicle_brand_Tata",
        "Vehicle_brand_Saab",
        "Vehicle_brand_SsangYong",
        "Vehicle_brand_Seat",
        "Vehicle_brand_Renault",
        "Vehicle_brand_Tesla",
        "Vehicle_brand_Suzuki",
        "Vehicle_brand_Smart",
        "Vehicle_brand_Skoda",
        "Vehicle_brand_Toyota",
        "Vehicle_brand_Syrena",
        "Vehicle_brand_Subaru",
        "Vehicle_brand_Volkswagen",
        "Vehicle_brand_Żuk",
        "Vehicle_brand_Volvo",
        "Vehicle_brand_RAM",
        "Vehicle_brand_Cupra",
        "Vehicle_brand_Abarth",
        "Vehicle_brand_Wartburg",
        "Vehicle_brand_Alpine",
        "Vehicle_brand_DS_Automobiles",
        "Vehicle_brand_Inny",
    ]
    df_pandas["brand_ratio"] = df_pandas[brand_columns].sum(axis=1) / len(brand_columns)

    # Production Year Category
    # year_columns = [
    #     "Year_of_production_2016",
    #     "Year_of_production_2012",
    #     "Year_of_production_2020",
    #     "Year_of_production_2019",
    #     "Year_of_production_2017",
    #     "Year_of_production_2014",
    #     "Year_of_production_1984",
    #     "Year_of_production_2013",
    #     "Year_of_production_2005",
    #     "Year_of_production_2000",
    #     "Year_of_production_2002",
    #     "Year_of_production_2018",
    #     "Year_of_production_2009",
    #     "Year_of_production_1995",
    #     "Year_of_production_2006",
    #     "Year_of_production_2004",
    #     "Year_of_production_2011",
    #     "Year_of_production_1992",
    #     "Year_of_production_2022",
    #     "Year_of_production_2008",
    #     "Year_of_production_1999",
    #     "Year_of_production_1994",
    #     "Year_of_production_2007",
    #     "Year_of_production_2023",
    #     "Year_of_production_2021",
    #     "Year_of_production_2015",
    #     "Year_of_production_1998",
    #     "Year_of_production_2001",
    #     "Year_of_production_2010",
    #     "Year_of_production_2003",
    #     "Year_of_production_1991",
    #     "Year_of_production_1987",
    #     "Year_of_production_1989",
    #     "Year_of_production_1961",
    #     "Year_of_production_1997",
    #     "Year_of_production_1996",
    #     "Year_of_production_1986",
    #     "Year_of_production_1993",
    #     "Year_of_production_1990",
    #     "Year_of_production_1982",
    #     "Year_of_production_1981",
    #     "Year_of_production_1957",
    #     "Year_of_production_1978",
    #     "Year_of_production_1974",
    #     "Year_of_production_1983",
    #     "Year_of_production_1985",
    #     "Year_of_production_1979",
    # ]
    # df_pandas["production_year_category"] = pd.cut(
    #     df_pandas[year_columns].sum(axis=1),
    #     bins=[1950, 1990, 2000, 2010, 2025],
    #     labels=["Older", "Mid-range", "Newer", "Future"],
    # )

    # Fuel Type Count
    fuel_columns = [
        "Fuel_type_Benzyna",
        "Fuel_type_Benzyna+LPG",
        "Fuel_type_Diesel",
        "Fuel_type_Elektryczny",
        "Fuel_type_Hybryda",
    ]
    df_pandas["fuel_type_ratio"] = df_pandas[fuel_columns].sum(axis=1) / len(
        fuel_columns
    )

    # Body Type Count
    body_columns = [
        "Body_type_Kabriolet",
        "Body_type_SUV",
        "Body_type_Sedan",
        "Body_type_Auta_małe",
        "Body_type_Coupe",
        "Body_type_Minivan",
        "Body_type_Kompakt",
        "Body_type_Auta_miejskie",
        "Body_type_Kombi",
    ]
    df_pandas["body_type_ratio"] = df_pandas[body_columns].sum(axis=1) / len(
        body_columns
    )

    # Door Count
    door_columns = [
        "Number_of_doors_3",
        "Number_of_doors_5",
        "Number_of_doors_4",
        "Number_of_doors_2",
        "Number_of_doors_6",
    ]
    df_pandas["door_number_ratio"] = df_pandas[door_columns].sum(axis=1) / len(
        door_columns
    )

    df_pandas.to_csv(
        "/home/konradballegro/orchestration/data/outputs/offers_filtered.csv",
        index=False,
    )

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

    # corr_matrix = X_train.corr(method="spearman").abs()
    # upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    # to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]
    # X_train = X_train.drop(to_drop, axis=1)
    # X_val = X_val.drop(to_drop, axis=1)
    # X_test = X_test.drop(to_drop, axis=1)

    X_train = X_train[selected_features]
    X_val = X_val[selected_features]
    X_test = X_test[selected_features]

    regressors = {}
    regressors.update({"XGBoost": XGBRegressor(n_jobs=-1)})

    parameters = {}

    # parameters.update({"XGBoost": {"n_estimators": [100, 200],
    #                             "max_depth": [3, 5, 8],
    #                             "learning_rate": [0.01, 0.1],
    #                             "subsample": [0.5, 0.8],
    #                             "colsample_bytree": [0.5, 0.8]}})

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

    # model = XGBRegressor()
    model.load_model("/home/konradballegro/orchestration/models/xgb copy.model")
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Mean Squared Error:", mse)
    print("R2:", r2)


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
