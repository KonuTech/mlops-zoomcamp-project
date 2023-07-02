import json
import logging
from functools import reduce
from typing import List

import mlflow
import pandas as pd
from flask import Flask, jsonify, request

CONFIG_PATH = "/home/konradballegro/scoring-event/config/config.json"


with open(CONFIG_PATH, encoding="UTF-8") as json_file:
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
COLUMNS_TO_ADD = config["COLUMNS_TO_ADD"]
SELECTED_FEATURES = config["SELECTED_FEATURES"]


# TRACKING_SERVER_HOST = "34.77.180.77"
RUN_ID = "ed549f18f6a64334b9873babbcb43dee"

# Configure logging
logging.basicConfig(
    filename="/home/konradballegro/scoring-event/app.log",
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


# Configure MLflow
# mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:5000")
logged_model = f"gs://mlops-zoomcamp/3/{RUN_ID}/artifacts/model"
model = mlflow.pyfunc.load_model(logged_model)

logging.debug("Model loaded: %s", logged_model)


app = Flask("otomoto-used-car-price-prediction")


# Oferta od,Kategoria,Pokaż oferty z numerem VIN,Ma numer rejestracyjny,Marka pojazdu,Model pojazdu,Wersja,Generacja,Rok produkcji,Przebieg,Pojemność skokowa,Rodzaj paliwa,Moc,Skrzynia biegów,Autonomia,Napęd,Pojemność baterii,Rodzaj własności baterii,Emisja CO2,Filtr cząstek stałych,Spalanie W Mieście,Typ nadwozia,Liczba drzwi,Liczba miejsc,Kolor,Metalik,Rodzaj koloru,Kierownica po prawej (Anglik),Kraj pochodzenia,Leasing,VAT marża,Faktura VAT,Okres gwarancji producenta,Możliwość finansowania,Pierwsza rejestracja,Zarejestrowany w Polsce,Pierwszy właściciel,Bezwypadkowy,Serwisowany w ASO,Stan,ABS,Apple CarPlay,Android Auto,Boczne poduszki powietrzne - tył,Boczna poduszka powietrzna kierowcy,CD,Centralny zamek,Elektryczne szyby przednie,Elektrycznie ustawiany fotel pasażera,Elektrycznie ustawiane lusterka,Immobilizer,Poduszka powietrzna kierowcy,Poduszka powietrzna pasażera,Radio fabryczne,Wspomaganie kierownicy,Alarm,Alufelgi,ASR (kontrola trakcji),Alarm ruchu poprzecznego z tyłu pojazdu,Asystent parkowania,Asystent hamowania awaryjnego w mieście,Asystent hamowania - Brake Assist,Aktywny asystent zmiany pasa ruchu,Asystent jazdy w korku,Aktywne rozpoznawanie znaków ograniczenia prędkości,Asystent (czujnik) martwego pola,Asystent świateł drogowych,Asystent zapobiegania kolizjom na skrzyzowaniu,Asystent pasa ruchu,Automatyczna kontrola zjazdu ze stoku,Boczne poduszki powietrzne - przód,Cyfrowy kluczyk,Czujnik deszczu,Czujnik martwego pola,Czujnik zmierzchu,Czujniki parkowania przednie,Czujniki parkowania tylne,Dach panoramiczny,Dach otwierany elektrycznie,Dostęp do internetu,Dynamiczne światła doświetlające zakręty,Dźwignia zmiany biegów wykończona skórą,Ekran dotykowy,Elektrochromatyczne lusterka boczne,Elektrochromatyczne lusterko wsteczne,Elektryczne szyby tylne,Elektrycznie ustawiane fotele,Elektryczny hamulec postojowy,Elektroniczny system rozdziału siły hamowania,Elektroniczna kontrola ciśnienia w oponach,Elektroniczna regul. charakterystyki zawieszenia,Elektrycznie ustawiany fotel kierowcy,ESP,Felgi aluminiowe od 21,Felgi aluminiowe 19,Fotele przednie wentylowane,Fotele przednie z funkcje masażu,Funkcja szybkiego ładowania,Gniazdo AUX,Gniazdo SD,Gniazdo USB,Hak,Hamulce z kompozytów ceramicznych,HUD (wyświetlacz przezierny),Interfejs Bluetooth,Isofix,Kabel do ładowania,Kamera panoramiczna 360,Kamera cofania,Kamera parkowania tył,Kamera w lusterku bocznym,Kierownica ze sterowaniem radia,Kierownica skórzana,Kierownica wielofunkcyjna,Kierownica sportowa,Kierownica ogrzewana,Klimatyzacja automatyczna,Klimatyzacja czterostrefowa,Klimatyzacja automatyczna: 3 strefowa,Klimatyzacja dwustrefowa,"Klimatyzacja automatyczna, dwustrefowa",Klimatyzacja automatyczna: 4 lub wiêcej strefowa,Klimatyzacja manualna,Klimatyzacja dla pasażerów z tyłu,Komputer pokładowy,Kontrola trakcji,Kontrola odległości od poprzedzającego pojazdu,Kontrola odległości z tyłu (przy parkowaniu),Kontrola odległości z przodu (przy parkowaniu),Kurtyny powietrzne - przód,Kurtyny powietrzne - tył,Keyless Go,Keyless entry,Lane assist - kontrola zmiany pasa ruchu,Lampy tylne w technologii LED,Lampy przednie w technologii LED,Lampy bi-ksenonowe,Lampy ksenonowe,Lampy doświetlające zakręt,Lampy przeciwmgielne,Lampy przeciwmgielne w technologii LED,Lusterka boczne ustawiane elektrycznie,Lusterka boczne składane elektrycznie,Ładowanie bezprzewodowe urządzeń,Łopatki zmiany biegów,MP3,Nawigacja GPS,Niezależny system parkowania,Odtwarzacz DVD,Ogranicznik prędkości,Ogrzewanie postojowe,Ogrzewane siedzenia tylne,Opony runflat,Opony letnie,Opony wielosezonowe,Oświetlenie drogi do domu,Oświetlenie wnętrza LED,Oświetlenie adaptacyjne,Park Assistant - asystent parkowania,Podgrzewana przednia szyba,Podgrzewane lusterka boczne,Podgrzewane przednie siedzenia,Podgrzewane tylne siedzenia,Podłokietniki - przód,Podłokietniki - tył,Poduszka powietrzna chroniąca kolana,Poduszki boczne przednie,Poduszki boczne tylne,Poduszka powietrzna centralna,Poduszka kolan kierowcy,Poduszka kolan pasażera,Poduszka powietrzna pasów bezpieczeństwa zm tyłu,Podgrzewany fotel pasażera,Przyciemniane szyby,Przyciemniane tylne szyby,Radio niefabryczne,Radio,Regulowane zawieszenie,Regul. elektr. podparcia lędźwiowego - kierowca,Regul. elektr. podparcia lędźwiowego - pasażer,Relingi dachowe,Rolety na bocznych szybach opuszczane ręcznie,Siedzenie z pamięcią ustawienia,Sportowe fotele - przód,Spryskiwacze reflektorów,Sterowanie funkcjami pojazdu za pomocą głosu,System minimalizujący skutki kolizji,System wspomagania hamowania,System ostrzegający o możliwej kolizji,System rekomendacji przerw podczas trasy,System powiadamiania o wypadku,System Start/Stop,System nawigacji satelitarnej,System odzyskiwania energii,System nagłośnienia,System hamowania awaryjnego dla ochrony pieszych,System rozpoznawania znaków drogowych,Szyberdach,Szyberdach szklany - przesuwny i uchylny elektrycz,Światła do jazdy dziennej,Światła LED,Światła do jazdy dziennej diodowe LED,Światła przeciwmgielne,Światła Xenonowe,Tapicerka skórzana,Tapicerka Alcantara,Tapicerka welurowa,Tempomat,Tempomat aktywny,Tempomat adaptacyjny ACC,Tuner TV,Uruchamianie silnika bez użycia kluczyków,Wielofunkcyjna kierownica,Wycieraczki,Wspomaganie ruszania pod górę- Hill Holder,Wyświetlacz typu Head-Up,Zmieniarka CD,Zawieszenie sportowe,Zawieszenie komfortowe,Zawieszenie powietrzne,Zawieszenie regulowane,Zewnętrzne oklejenie,Zestaw głośnomówiący,Zmiana biegów w kierownicy,Cena,Szczegóły ceny,Waluta,url_path,id,epoch
# Osoby prywatnej,Osobowe,Tak,Tak,Mercedes-Benz,Klasa V,250 d 4-Matic Exclusive 7G-Tronic,II (2014-),2016,110 000 km,2 143 cm3,Diesel,190 KM,Automatyczna,,4x4 (dołączany automatycznie),,,,,7 l/100km,Minivan,4,7,Biały,,Metalik,,Polska,,,,,,,Tak,,Tak,,Używane,1.0,,,,1.0,,,1.0,1.0,,,1.0,1.0,,1.0,,,,,,1.0,1.0,,,1.0,1.0,1.0,,,1.0,1.0,,1.0,,1.0,,,,,,,,1.0,,,1.0,,1.0,1.0,1.0,,1.0,1.0,,,1.0,,,,,1.0,,,,1.0,,,1.0,,1.0,1.0,1.0,1.0,1.0,,,1.0,,,,,,,1.0,,1.0,1.0,1.0,1.0,1.0,1.0,1.0,,1.0,1.0,1.0,,,1.0,1.0,,1.0,1.0,,,,,,,1.0,,1.0,,,,,1.0,1.0,1.0,,1.0,,,1.0,1.0,,,,,1.0,,,1.0,,1.0,,1.0,,1.0,,,,1.0,,,,,1.0,1.0,,,1.0,1.0,,1.0,,1.0,,1.0,1.0,,1.0,,,1.0,,,1.0,,,,,,1.0,1.0,,,1.0,1.0,,,,1.0,1.0,225000,Do negocjacji,PLN,https://www.otomoto.pl/oferta/mercedes-benz-klasa-v-mercedes-v250d-2016-niski-przebieg-doskonaly-stan-4x4-ID6Fqn8p.html,a216c3aff82b3e8fc6b2644539c9196c21ccb8ff,1688290322


def data_read():
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
    file_path = "/home/konradballegro/data/inputs/mercedes-benz_copy.csv"
    logging.info(f"Reading data from file: {file_path}")
    # return (
    #     spark_session.read.option("header", header)
    #     .option("inferSchema", infer_schema)
    #     .csv(file_path)
    # )

    df = pd.read_csv(file_path)

    return df


def data_filter(df):
    """
    Filters the data based on specified conditions.

    Args:
        df (DataFrame): The input Spark DataFrame.

    Returns:
        filtered_df (DataFrame): The filtered Spark DataFrame.

    """
    logging.info("Filtering data...")
    conditions = [
        (df["Currency"] == "PLN"),
        (df["Country of origin"] == "Polska"),
        (df["Accident-free"].notnull()),
        (df["Price"].notnull()),
        (df["Offer from"].notnull()),
        (df["Condition"].notnull()),
        (df["Condition"] == "Używane"),
        (df["Vehicle brand"].notnull()),
        (df["Year of production"].notnull()),
        (df["Mileage"].notnull()),
        (df["Fuel type"].notnull()),
        (df["Power"].notnull()),
        (df["Gearbox"].notnull()),
        (df["Body type"].notnull()),
        (df["Number of doors"].notnull()),
    ]
    logging.info("Data filtered")
    return df[reduce(lambda a, b: a & b, conditions)]


def data_preprocess(df):
    """
    Preprocesses the data by cleaning and transforming the columns.

    Args:
        df (DataFrame): The input Spark DataFrame.

    Returns:
        cleaned_df (DataFrame): The preprocessed Spark DataFrame.

    """
    logging.info("Preprocessing data...")
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
        .astype(float)
    )
    df_cleaned["Fuel type"] = df["Fuel type"]
    df_cleaned["Power"] = (
        df["Power"].astype(str).str.replace(" ", "").str.replace("KM", "").astype(int)
    )
    df_cleaned["Gearbox"] = df["Gearbox"]
    df_cleaned["Body type"] = df["Body type"]
    df_cleaned["Number of doors"] = (
    df["Number of doors"].astype(str).str.replace("\.0$", "")
    )
    df_cleaned["URL path"] = df["URL path"]
    df_cleaned["ID"] = df["ID"]
    df_cleaned["Epoch"] = df["Epoch"]

    df.to_csv(
        "/home/konradballegro/scoring-event/test_data_preprocess.csv", index=False
    )

    logging.info("Data preprocessed")
    return df_cleaned


def features_engineer(
    df,
    distinct_columns: List[str],
    columns_to_drop: List[str],
    columns_to_add: List[str],
    selected_features: List[str]
):
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
    logging.info("Performing feature engineering...")
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

            # Assign the dummy variable to the new column
            df[column_name] = dummy_variable

    
    # Check if each column exists in the DataFrame
    columns_to_add = [col for col in columns_to_add if col not in df.columns]

    # Add the missing columns with default value 0 using pd.concat()
    df = pd.concat([df, pd.DataFrame(0, index=df.index, columns=columns_to_add)], axis=1)

    # existing_columns = set(df.columns)
    # columns_to_drop = [col for col in columns_to_drop if col in existing_columns]
    df = df.drop(columns_to_drop, axis=1)
    df = df.dropna(subset=["Price"])

    # Price per Mileage
    df["price_per_mileage"] = df["Price"] / df["Mileage"]

    # Power-to-Price Ratio
    df["power_to_price_ratio"] = df["Power"] / df["Price"]

    df = df[selected_features]
    df.to_csv(
        "/home/konradballegro/scoring-event/test_features_engineer.csv", index=False
    )

    logging.info("Feature engineering completed")
    return df.to_numpy()


# Transforming original features here:
def prepare_features(car):
    # Scrap offers.csv file

    # Create Spark session
    # session_create(
    #     spark_session_scope=SPARK_SESSION_SCOPE,
    #     spark_session_name=SPARK_SESSION_NAME,
    # )

    # Read data
    data_input = data_read()

    # Filter data
    data_filtered = data_filter(df=data_input)

    # Preprocess data
    data_preprocessed = data_preprocess(df=data_filtered)

    features = features_engineer(
        df=data_preprocessed,
        distinct_columns=DISTINCT_COLUMNS,
        columns_to_drop=COLUMNS_TO_DROP,
        columns_to_add=COLUMNS_TO_ADD,
        selected_features=SELECTED_FEATURES
    )

    # features = {
    #     "Mileage": 1.0,
    #     "Power": 1.0,
    #     "Vehicle_brand_Jaguar": 1.0,
    #     "Vehicle_brand_Kia": 1.0,
    #     "Vehicle_brand_Lamborghini": 1.0,
    #     "Vehicle_brand_Mercedes-Benz": 1.0,
    #     "Vehicle_brand_Mazda": 1.0,
    #     "Vehicle_brand_MINI": 1.0,
    #     "Vehicle_brand_Skoda": 1.0,
    #     "Vehicle_brand_Volvo": 1.0,
    #     "Vehicle_brand_Inny": 1.0,
    #     "Year_of_production_2020": 1.0,
    #     "Year_of_production_2017": 1.0,
    #     "Year_of_production_2014": 1.0,
    #     "Year_of_production_2013": 1.0,
    #     "Year_of_production_2000": 1.0,
    #     "Year_of_production_2009": 1.0,
    #     "Year_of_production_2006": 1.0,
    #     "Year_of_production_2022": 1.0,
    #     "Year_of_production_1999": 1.0,
    #     "Year_of_production_1998": 1.0,
    #     "Year_of_production_1997": 1.0,
    #     "Fuel_type_Benzyna": 1.0,
    #     "Fuel_type_Benzyna+LPG": 1.0,
    #     "Fuel_type_Diesel": 1.0,
    #     "Fuel_type_Hybryda": 1.0,
    #     "Gearbox_Manualna": 1.0,
    #     "Body_type_Kabriolet": 1.0,
    #     "Body_type_Kompakt": 1.0,
    #     "Body_type_Kombi": 1.0,
    #     "Number_of_doors_5": 1.0,
    #     "Number_of_doors_4": 1.0,
    #     "Number_of_doors_2": 1.0,
    #     "price_per_mileage": 1.0,
    #     "power_to_price_ratio": 1.0,
    # }

    # Create a DataFrame with a single row
    # return pd.DataFrame(features, index=[0])
    return features


def predict(features):
    preds = model.predict(features)
    return float(preds[0])


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    try:
        car = request.get_json()
        logging.debug("Received JSON data: %s", car)

        features = prepare_features(car)
        pred = predict(features)

        result = {"price": pred, "model_version": RUN_ID}

        return jsonify(result)
    except Exception as e:
        logging.exception("Prediction failed")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=9696)
