import os

from modules.scrapers.offers_scraper import ManufacturerScraper
from google.cloud import storage
from prefect import flow, task


# scrap data directly to GCP bucket


@task(retries=0, retry_delay_seconds=2)
def scrap_data(manufacturers_file, destination_path):
    scraper = ManufacturerScraper(
        path_manufacturers_file=manufacturers_file, path_data_directory=destination_path
    )
    scraper.scrap_all_manufacturers()
    scraper.dump_data()


@task(retries=0, retry_delay_seconds=2)
def upload_directory_to_bucket(bucket_name, source_directory, destination_directory):
    """Uploads a directory to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    for root, dirs, files in os.walk(source_directory):
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
    manufacturers_file = "manufacturers.txt"
    bucket_name = "mlops-zoomcamp"
    data_directory = "data"

    scrap_data(
        manufacturers_file=manufacturers_file, destination_path=f"{data_directory}/"
    )

    upload_directory_to_bucket(
        bucket_name=bucket_name,
        source_directory="/home/konradballegro/orchestration/data/",
        destination_directory=f"{data_directory}/training/",
    )


if __name__ == "__main__":
    otomoto_scraping_flow()
