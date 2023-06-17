import os
import logging
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import mlflow
from flask import Flask, request, jsonify

TRACKING_SERVER_HOST = "34.77.180.77"
RUN_ID = '8e97a22aff684e69aaabdb7cf04fec30'

# Configure logging
logging.basicConfig(filename="app.log", level=logging.DEBUG,
                    format="%(asctime)s - %(levelname)s - %(message)s")

# Configure MLflow
mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:5000")
logged_model = f'gs://mlops-zoomcamp/2/{RUN_ID}/artifacts/models_mlflow'
model = mlflow.pyfunc.load_model(logged_model)
logging.debug("Model loaded: %s", logged_model)

app = Flask('duration-prediction')


def prepare_features(ride):
    features = {
        'PU_DO': f"{ride['PULocationID']}_{ride['DOLocationID']}",
        'trip_distance': ride['trip_distance']
    }

    # Create a DataFrame with a single row
    features_df = pd.DataFrame(features, index=[0])

    # Encode 'PU_DO' column as categorical
    label_encoder = LabelEncoder()
    features_df['PU_DO'] = label_encoder.fit_transform(features_df['PU_DO'])

    return features_df


def predict(features):
    preds = model.predict(features)
    return float(preds[0])


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    try:
        ride = request.get_json()
        logging.debug("Received JSON data: %s", ride)

        features = prepare_features(ride)
        pred = predict(features)

        result = {
            'duration': pred,
            'model_version': RUN_ID
        }

        return jsonify(result)
    except Exception as e:
        logging.exception("Prediction failed")
        return jsonify({'error': str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=9696)
