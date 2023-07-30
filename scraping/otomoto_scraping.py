"""
Otomoto Scraping Script

This script contains a Prefect flow that scrapes data
for car manufacturers' names from a file,
performs web scraping on www.otomoto.pl
to obtain car offers data for each manufacturer,
and then uploads the scraped data to a Google Cloud Storage bucket.

Tasks:
    scrap_data: Task that scrapes data for car manufacturers names
    and saves it to the specified destination path.
    upload_directory_to_bucket: Task that uploads a directory
    to a specified Google Cloud Storage bucket.

Prefect Flow:
    otomoto_scraping_flow: Prefect flow that defines the data scraping and uploading process.
"""

import os

from google.cloud import storage
from prefect import flow, task

from scrapers.offers_scraper import ManufacturerScraper


@task(retries=0, retry_delay_seconds=2)
def scrap_data(manufacturers_file: str):
    """
    Scrapes data for car manufacturers names and saves it to the specified destination path.

    Args:
        manufacturers_file (str): The path to the file containing car manufacturers names.
    """
    scraper = ManufacturerScraper(path_manufacturers_file=manufacturers_file)
    scraper.scrap_all_manufacturers()
    scraper.dump_data()


@task(retries=0, retry_delay_seconds=2)
def upload_directory_to_bucket(
    bucket_name: str, source_directory: str, destination_directory: str
):
    """
    Uploads a directory to the specified bucket.

    Args:
        bucket_name (str): The name of the bucket.
        source_directory (str): The path to the source directory to be uploaded.
        destination_directory (str): The destination directory
        in the bucket where the files will be uploaded.
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    for root, _, files in os.walk(source_directory):
        for filename in files:
            local_file_path = os.path.join(root, filename)
            blob_path = os.path.join(
                destination_directory,
                os.path.relpath(local_file_path, source_directory),
            )

            blob = bucket.blob(blob_path)
            blob.upload_from_filename(local_file_path)

            print(f"File {local_file_path} uploaded to {blob_path}.")


@flow
def otomoto_scraping_flow():
    """
    Prefect flow that orchestrates the data scraping
    and uploading process for car manufacturers' data.

    This flow performs the following tasks:
    1. scrap_data: Task that scrapes data for car manufacturers names
    and saves it to the specified destination path.
    2. upload_directory_to_bucket: Task that uploads a directory
    to a specified Google Cloud Storage bucket.

    The flow starts by calling the scrap_data task to scrape data
    for car manufacturers from a file.
    Once the data is scraped, it is saved to the specified destination path.

    Next, the flow calls the upload_directory_to_bucket task
    to upload the scraped data directory
    to a Google Cloud Storage bucket with the specified bucket name
    and destination directory.

    This flow is designed to be used with Prefect, a dataflow automation framework.
    """
    manufacturers_file = "manufacturers.txt"
    bucket_name = "mlops-zoomcamp"
    data_directory = "data"

    scrap_data(manufacturers_file=manufacturers_file)

    upload_directory_to_bucket(
        bucket_name=bucket_name,
        source_directory="/home/konradballegro/data/raw/",
        destination_directory=f"{data_directory}/raw/",
    )


if __name__ == "__main__":
    otomoto_scraping_flow()
