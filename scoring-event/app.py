import logging

import mlflow
import pandas as pd
from flask import Flask, jsonify, request

# TRACKING_SERVER_HOST = "34.77.180.77"
RUN_ID = "ed549f18f6a64334b9873babbcb43dee"

# Configure logging
logging.basicConfig(
    filename="scoring.log",
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


# Configure MLflow
# mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:5000")
logged_model = f"gs://mlops-zoomcamp/3/{RUN_ID}/artifacts/model"
model = mlflow.pyfunc.load_model(logged_model)
logging.debug("Model loaded: %s", logged_model)


app = Flask("otomoto-used-car-price-prediction")


def prepare_features(car):
    features = {
        "Mileage": 1.0,
        "Power": 1.0,
        "Vehicle_brand_Jaguar": 1.0,
        "Vehicle_brand_Kia": 1.0,
        "Vehicle_brand_Lamborghini": 1.0,
        "Vehicle_brand_Mercedes-Benz": 1.0,
        "Vehicle_brand_Mazda": 1.0,
        "Vehicle_brand_MINI": 1.0,
        "Vehicle_brand_Skoda": 1.0,
        "Vehicle_brand_Volvo": 1.0,
        "Vehicle_brand_Inny": 1.0,
        "Year_of_production_2020": 1.0,
        "Year_of_production_2017": 1.0,
        "Year_of_production_2014": 1.0,
        "Year_of_production_2013": 1.0,
        "Year_of_production_2000": 1.0,
        "Year_of_production_2009": 1.0,
        "Year_of_production_2006": 1.0,
        "Year_of_production_2022": 1.0,
        "Year_of_production_1999": 1.0,
        "Year_of_production_1998": 1.0,
        "Year_of_production_1997": 1.0,
        "Fuel_type_Benzyna": 1.0,
        "Fuel_type_Benzyna+LPG": 1.0,
        "Fuel_type_Diesel": 1.0,
        "Fuel_type_Hybryda": 1.0,
        "Gearbox_Manualna": 1.0,
        "Body_type_Kabriolet": 1.0,
        "Body_type_Kompakt": 1.0,
        "Body_type_Kombi": 1.0,
        "Number_of_doors_5": 1.0,
        "Number_of_doors_4": 1.0,
        "Number_of_doors_2": 1.0,
        "price_per_mileage": 1.0,
        "power_to_price_ratio": 1.0,
    }

    # Create a DataFrame with a single row
    return pd.DataFrame(features, index=[0])


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
