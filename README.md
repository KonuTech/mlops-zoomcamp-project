# Used car price prediction

## Objective

This repository contains the final project for the [MLOps Zoomcamp](https://github.com/DataTalksClub/mlops-zoomcamp) course provided by [DataTalks.Club](https://datatalks.club/).

The goal of the project is to apply what has been learned during the MLOps Zoomcamp. This project aims at building an end-to-end machine learning system to predict the prices of used cars based on a selection of available attributes.

## Dataset

The dataset used to feed the MLOps pipeline has been scraped from [otomoto.pl](https://www.otomoto.pl/). It contains data of used cards offers from the following [manufacturers](https://github.com/KonuTech/mlops-zoomcamp-project/blob/main/data/metadata/manufacturers.txt) . The dataset is updated (scraped) weekly and is characterized by the following [features](https://github.com/KonuTech/mlops-zoomcamp-project/blob/main/data/metadata/header_en.txt) . The data used for training is available at the following [public GCP URL](https://storage.googleapis.com/mlops-zoomcamp/data/training/offers.csv) (offers.csv file of 85MB). Before training the data was [cleaned and preprocessed](https://github.com/KonuTech/mlops-zoomcamp-project/blob/main/training/otomoto_training.py). Feature enginering was also applied.

## MLOps pipeline

### Applied technologies

| Name | Scope |
| --- | --- |
| Google Compute Engine | Remote processing units |
| Google Cloud Storage | Storage space for data and trained models |
| Jupyter Notebooks | Exploratory data analysis and pipeline prototyping |
| PySpark | Data preprocessing |
| Pandas | Feature engineering |
| Scikit-learn | Training pipeline, including Feature selection |
| XGBoost | Regressor |
| Prefect | Workflow orchestration |
| MLFlow | Experiment tracking and model registry |
| PostgreSQL | MLFLow experiment tracking database |
| Flask | Web server |
| FastAPI | Web server |
| EvidentlyAI | ML models evaluation and monitoring |
| pytest | Python unit testing suite |
| pylint | Python static code analysis |
| black | Python code formatting |
| isort | Python import sorting |
| Pre-Commit Hooks | Before submission code issue identification |

### Architecture

TODO: a high-level schema of an architecture

### Steps to run the project

The MLOps pipeline is fully dockerised and can be easily deployed via the following steps:

1. Clone the [mlops-zoomcamp-project](https://github.com/KonuTech/mlops-zoomcamp-project) repository:

    ```bash
    $ git clone https://github.com/KonuTech/mlops-zoomcamp-project.git
    ```

2. Install the pre-requisites necessary to run the pipeline:

    ```bash
    $ cd mlops-zoomcamp-project
    $ sudo apt install make
    $ make setup
    ```

6. Launch the MLOps pipeline:

    ```
    $ make run
    ```

    Once ready, the following services will be available:

    | Service | Port | Interface | Description |
    | --- | --- | --- | --- |
    | Web Application | 80 | 0.0.0.0 | Prediction web service (see picture below) |
    | Prefect | 4200 | 127.0.0.1 | Training workflow orchestration |
    | MLFlow | 5000 | 127.0.0.1 | Experiment tracking and model registry |
    | MinIO | 9001 | 127.0.0.1 | S3-equivalent bucket management |
    | Evidently | 8085 | 127.0.0.1 | Data and target drift report generation (`/dashboard` route) |
    | Grafana | 3000 | 127.0.0.1 | Data and target drift real-time dashboards |


    <img src="images/webservice.png" width="100%"/>


### Training

Once the MLOps pipeline has been started, the prediction web service can already work thanks to a default pre-trained model available in the Docker image. In order to enable pipeline training workflow it is necessary to create a scheduled Prefect deployment via:

```
$ make deployment
```

The training workflow will be then automatically executed every day. It will download the latest dataset (if the Kaggle credentials have been provided), search the best model in terms of accuracy among XGBoost, Support Vector Machine and Random Forest and finally will store it in the model registry. It is worth noting the training workflow can also be immediately executed without waiting the next schedule:

```
$ make train
```

Once the updated model is ready, it can be moved to production by restarting the pipeline:

```
$ make restart
```

the web service will automatically connect to the registry and get the most updated model. If the model is still not available, it will continue to use the default one.


### Monitoring

It is possible to generate simulated traffic via:

```
$ make generate-traffic
```

Then, the prediction service can be monitored via:

- Grafana (in real-time): `http://127.0.0.1:3000`
- Evidently (for report generation): `http://127.0.0.1:8085/dashboard`


------------

Project Structure
------------
    ├── Makefile
    ├── Pipfile
    ├── README.md
    ├── data
    │   ├── metadata
    │   │   ├── header_en.txt
    │   │   ├── header_pl.txt
    │   │   ├── manufacturers.txt
    │   │   └── manufacturers_batch.txt
    │   ├── preprocessed
    │   │   ├── offers_filtered.csv
    │   │   └── offers_preprocessed.csv
    │   ├── raw
    │   │   ├── abarth.csv
    │   │   ├── acura.csv
    │   │   ├── aixam.csv
    │   │   ├── nissan.csv
    │   │   └── offers.csv
    │   ├── scored
    │   │   ├── offers_scored.csv
    │   │   ├── offers_scored_current.csv
    │   │   └── offers_scored_reference.csv
    │   └── training
    │       └── offers.csv
    ├── models
    │   └── xgb.model
    ├── monitoring
    │   ├── config
    │   │   └── config.py
    │   ├── otomoto_monitoring.py
    │   ├── reports
    │   │   ├── model_performance.html
    │   │   └── target_drift.html
    │   └── src
    │       └── utils
    │           ├── data.py
    │           └── reports.py
    ├── notebooks
    │   ├── explainer_xgb.ipynb
    │   ├── outputs
    │   │   └── reports
    │   │       ├── profiling_filtered.html
    │   │       └── xgb_explainer.html
    │   ├── profiling.ipynb
    │   └── spark_test.ipynb
    ├── otomoto_scraping_flow-deployment.yaml
    ├── otomoto_training_flow-deployment.yaml
    ├── projects_tree.txt
    ├── requirements.txt
    ├── scoring_batch
    │   ├── __init__.py
    │   ├── app.log
    │   ├── app.py
    │   ├── config
    │   │   └── config.json
    │   └── otomoto_scoring_batch.py
    ├── scraping
    │   ├── logs
    │   │   └── app.log
    │   ├── otomoto_scraping.py
    │   ├── scrapers
    │   │   ├── __init__.py
    │   │   ├── get_offers.py
    │   │   └── offers_scraper.py
    │   └── utils
    │       └── logger.py
    ├── streamlit
    │   ├── app.py
    │   ├── static
    │   │   └── logo.png
    │   └── utils
    │       └── ui.py
    ├── tests
    │   ├── __init__.py
    │   ├── config
    │   │   └── config.json
    │   ├── data
    │   │   ├── preprocessed
    │   │   │   ├── nissan_preprocessed.csv
    │   │   │   └── offers_preprocessed.csv
    │   │   ├── raw
    │   │   │   └── nissan.csv
    │   │   └── scored
    │   │       └── offers_scored.csv
    │   └── model_test.py
    ├── training
    │   ├── config
    │   │   └── config.json
    │   └── otomoto_training.py
    └── tree.txt

### The space for an improvement:
* Containerization of all apps
* CI/CD techniques
* Terraform
* Data engineering techniques for maintaining scraped data
* A dashboard where users can input values for a prediction
* Retraining of a model if any drifts are detected

I will add the above improvements along with the next iterations of the [Machine Learning Zoomcamp](https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/course-zoomcamp) and [Data Engineering Zoomcamp](https://github.com/DataTalksClub/data-engineering-zoomcamp) courses from [DataTalks.Club](https://datatalks.club/). The learning is not stopping here.

### Peer review criterias - a self assassment:
* Problem description
    * 2 points: The problem is well described and it's clear what the problem the project solves
* Cloud
    * 2 points: The project is developed on the cloud OR uses localstack (or similar tool) OR the project is deployed to Kubernetes or similar container management platforms
* Experiment tracking and model registry
    * 4 points: Both experiment tracking and model registry are used
* Workflow orchestration
    * 4 points: Fully deployed workflow 
* Model deployment
    * 2 points: Model is deployed but only locally
* Model monitoring
    * 2 points: Basic model monitoring that calculates and reports metrics
* Reproducibility
    * 4 points: Instructions are clear, it's easy to run the code, and it works. The versions for all the dependencies are specified.
* Best practices
    * There are unit tests (1 point)
    * Linter and/or code formatter are used (1 point)
    * There's a Makefile (1 point)
    * There are pre-commit hooks (1 point)
