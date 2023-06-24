# prefect deployment build -n mlops-gs-test -p default-agent-pool -q mlops-gs orchestrate_gs.py:main_flow_gs

import os

from modules.scrapers.offers_scraper import ManufacturerScraper
from google.cloud import storage
from prefect import flow, task


# scrap data directly to GCP bucket

@task(retries=0, retry_delay_seconds=2)
def scrap_data(bucket_name, manufacturers_file, destination_path):
    scraper = ManufacturerScraper(path_manufacturers_file=manufacturers_file, path_data_directory=destination_path)
    scraper.scrap_all_manufacturers()
    scraper.dump_data()
    
    # source_path = os.path.join(scraper.path_data_directory)
    # destination_path = os.path.join("data")
    
    # storage_client = storage.Client()
    # bucket = storage_client.bucket(bucket_name)
    # blob = bucket.blob(destination_path)
    # blob.upload_from_filename(source_path)


@task(retries=0, retry_delay_seconds=2)
def upload_directory_to_bucket(bucket_name, source_directory, destination_directory):
    """Uploads a directory to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    for root, dirs, files in os.walk(source_directory):
        for filename in files:
            local_file_path = os.path.join(root, filename)
            blob_path = os.path.join(destination_directory, os.path.relpath(local_file_path, source_directory))

            blob = bucket.blob(blob_path)
            blob.upload_from_filename(local_file_path)

            print(f"File {local_file_path} uploaded to {blob_path}.")


# @task(retries=3, retry_delay_seconds=2)
# def upload_header_to_bucket(bucket_name, source_path, destination_path):
#     storage_client = storage.Client()
#     bucket = storage_client.bucket(bucket_name)
#     blob = bucket.blob(destination_path)
#     blob.upload_from_filename(source_path)



# copy data from GCP bucket to VM

# read data

# preprocess data with spark

# scale two columns

# add feature engineering, including interactions between continues variables 

# normalize skewed target

# split data into train, val, test

# train and save xgb model using MlFlow and preselected hyperparameters

# define main flow and tracking server


@flow
def otomoto_training_flow():

    header_file = "header_en.txt"
    manufacturers_file = "manufacturers.txt"
    bucket_name = "mlops-zoomcamp"
    data_directory = "data"

    scrap_data(
        bucket_name=bucket_name,
        manufacturers_file=manufacturers_file,
        destination_path=f"{data_directory}/"
    )

    upload_directory_to_bucket(
        bucket_name=bucket_name,
        source_directory=f"/home/konradballegro/orchestration/data/",
        destination_directory=f"{data_directory}/training/"
        )

    # upload_header_to_bucket(
    #     bucket_name=bucket_name,
    #     source_path=f"{data_directory}/inputs/{header_file}",
    #     destination_path=f"{data_directory}/header_en.txt"
    #     )


if __name__ == '__main__':
    print("hello world")
    otomoto_training_flow()
