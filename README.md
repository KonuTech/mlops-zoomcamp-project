# Used car price prediction

## Objective

This repository contains the final project for the [MLOps Zoomcamp](https://github.com/DataTalksClub/mlops-zoomcamp) course provided by [DataTalks.Club](https://datatalks.club/).

The goal of the project is to apply what has been learned during the MLOps Zoomcamp. This project aims at building an end-to-end machine learning system to predict the prices of used cars based on a selection of available attributes.

## Dataset

The dataset used to feed the MLOps pipeline has been scraped from [otomoto.pl](https://www.otomoto.pl/). It contains data of used car offers from the following [manufacturers](https://github.com/KonuTech/mlops-zoomcamp-project/blob/main/data/metadata/manufacturers.txt). The dataset is updated (scraped) weekly and is characterized by the following [features](https://github.com/KonuTech/mlops-zoomcamp-project/blob/main/data/metadata/header_en.txt). The data used for training is available at the following [public GCP URL](https://storage.googleapis.com/mlops-zoomcamp/data/training/offers.csv) (offers.csv file of 85MB). Before training the data was [cleaned and preprocessed](https://github.com/KonuTech/mlops-zoomcamp-project/blob/main/training/otomoto_training.py). Feature enginering was also applied.

## MLOps pipeline

### Applied technologies

| Name | Scope |
| --- | --- |
| Google Compute Engine | Remote processing units |
| Google Cloud Storage Bucket | Storage space for data and trained models |
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

At the moment, the MLOps pipeline is not dockerized. The work over the project's content will continue during [Machine Learning Zoomcamp](https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/course-zoomcamp) and [Data Engineering Zoomcamp](https://github.com/DataTalksClub/data-engineering-zoomcamp) courses from [DataTalks.Club](https://datatalks.club/):

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

6. Run services (details below):

    | Service | Port | Interface | Description |
    | --- | --- | --- | --- |
    | Prefect | 4200 | 127.0.0.1 | Data scraping and training workflow orchestration |
    | MLFlow | 5000 | 0.0.0.0 | Experiment tracking and model registry |
    | Flask Web Application | 80 | 0.0.0.0 | Batch prediction web service |
    | Fast API Web Application | 80 | 0.0.0.0 | Batch prediction web service |
    | Evidently | 8085 | 127.0.0.1 | Data drift and target drift report generation |


<img src="images/webservice.png" width="60%"/>

### Data scraping
Launch Prefect Server:

    ```
    $ prefect server start
    ```
Trigger the process manually or wait until the scheduled one will start:

    ```
    $ python3 .scraping/otomoto_scraping.py
    ```
On my side, the process of data scraping is scheduled to start each Saturday at 10:00 AM.

### Training

Create a Google Cloud Storage Bucket for training artifacts and create a Tracking Server with the help of the following [blog](https://kargarisaac.github.io/blog/mlops/data%20engineering/2022/06/15/MLFlow-on-GCP.html) .

Run your VM instances:

<img src="static/tracking_server_03.jpg" width="60%"/>

Lunch your Mlflow Tracking server (don't mind the credentials - the project is turned off):

    ```bash
    $ mlflow server -h 0.0.0.0 -p 5000
    --backend-store-uri postgresql://admin:admin@10.74.208.3:5432/mlflow_db
    --default-artifact-root gs://mlops-zoomcamp
    ```

<img src="static/tracking_server_01.jpg" width="60%"/>

Check if your MLFlow app is up and runing:

<img src="static/tracking_server_04.jpg" width="60%"/>

Now launch Prefect Server:

```
$ prefect server start
```

Trigger the process manually or wait until the scheduled one starts. The training process uses data previously scraped, which is now stored on Google Cloud Storage Bucket:

    ```
    $ python3 .training/otomoto_training.py
    ```

Below screen shows successful training run:

<img src="static/tracking_server_06.jpg" width="60%"/>

On my side, the process of training is scheduled to start each Monday at 10:00 AM. By default, the GUI of Prefect marks scheduled instances with yellow dots:

<img src="static/tracking_server_05.jpg" width="60%"/>

Once the updated model is ready, it can be moved to production by MLOps model registry. All training artifacts are stored on Google Cloud Storage Bucket.

<img src="static/tracking_server_02.jpg" width="60%"/>

### Batch prediction

To perform Batch prediction using a model stored as an artifact on Google Cloud Storage Bucket, we first need to run the Flask app:
```
$ python3 ./scoring_batch/app.py
```
Now, we can perform a batch scoring. Run:
```
$ python3 ./scoring_batch/otomoto_scoring_batch.py
```
At the moment, I use a file ./data/metadata/manufacturers.txt as a means to provide the name of a manufacturer for:
 * scraping a new batch of data from otomoto.pl
 * scoring that data by applying pre-trained model.

The new batch of scraped and scored data can be used  to produce reports of Target Drift and Data Drift.

### Monitoring

For monitoring purposes I am using Evidently AI. At the moment I am tracking [Target Drift](https://github.com/KonuTech/mlops-zoomcamp-project/blob/main/monitoring/reports/model_performance.html) and [Data Drift](https://github.com/KonuTech/mlops-zoomcamp-project/blob/main/monitoring/reports/model_performance.html) (use the links to download static .html reports). First, to create reports run FastAPI app:
```
$ cd ./monitoring
$ uvicorn otomoto_monitoring:app
```
Now, launch Streamlit app:
```
$ cd ./streamlit
$ streamlit run ./app.py
```
Check the app under 127.0.0.1:8501:

<img src="static/tracking_server_07.jpg" width="60%"/>

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

### The space for improvement includes:
* Containerization of all apps
* CI/CD techniques
* Terraform
* Data engineering techniques for maintaining scraped data
* A dashboard where users can input values for a prediction
* Retraining of a model if any drifts are detected

I will add the above improvements along with the next iterations of the [Machine Learning Zoomcamp](https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/course-zoomcamp) and [Data Engineering Zoomcamp](https://github.com/DataTalksClub/data-engineering-zoomcamp) courses from [DataTalks.Club](https://datatalks.club/). The learning does not stop here.

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
