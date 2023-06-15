import mlflow
import os

TRACKING_SERVER_HOST = "34.77.180.77"
mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:5000")

print(f"tracking URI: '{mlflow.get_tracking_uri()}'")

from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

mlflow.set_experiment("my-experiment-1")

with mlflow.start_run():

    X, y = load_iris(return_X_y=True)

    params = {"C": 0.1, "random_state": 42}
    mlflow.log_params(params)

    lr = LogisticRegression(**params).fit(X, y)
    y_pred = lr.predict(X)
    mlflow.log_metric("accuracy", accuracy_score(y, y_pred))

    mlflow.sklearn.log_model(lr, artifact_path="models")
    print(f"default artifacts URI: '{mlflow.get_artifact_uri()}'")

mlflow.search_experiments()
