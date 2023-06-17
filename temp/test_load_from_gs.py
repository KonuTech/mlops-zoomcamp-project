import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/konradballegro/.ssh/ny-rides-konrad-b5129a6f2e66.json"
from google.cloud import storage

def download_artifacts(bucket_name, artifact_path, destination_path):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(artifact_path)
    blob.download_to_filename(destination_path)

# Example usage:
bucket_name ='mlops-zoomcamp'
artifact_path = 'data/green_tripdata_2021-01.parquet'
destination_path = "/home/konradballegro/orchestration/_test/green_tripdata_2021-01.parquet"

download_artifacts(bucket_name, artifact_path, destination_path)
