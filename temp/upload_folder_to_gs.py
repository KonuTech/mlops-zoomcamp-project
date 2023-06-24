import argparse
import os
from google.cloud import storage


def upload_blob(bucket_name, source_folder_path, destination_blob_name):
    """Uploads a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    for root, dirs, files in os.walk(source_folder_path):
        for filename in files:
            # get the path of the file in local filesystem
            local_file_path = os.path.join(root, filename)

            # construct the path of the blob in the bucket
            blob_path = os.path.join(
                destination_blob_name,
                os.path.relpath(local_file_path, source_folder_path),
            )

            # create a new blob and upload the file's content
            blob = bucket.blob(blob_path)
            blob.upload_from_filename(local_file_path)

            print(f"File {local_file_path} uploaded to {blob_path}.")


def main():
    # Create the parser
    parser = argparse.ArgumentParser(
        description="Upload a folder to Google Cloud Storage"
    )

    # Add the arguments
    parser.add_argument(
        "Bucket", metavar="bucket", type=str, help="The name of the bucket"
    )
    parser.add_argument(
        "Source", metavar="source", type=str, help="The path of the source folder"
    )
    parser.add_argument(
        "Destination",
        metavar="destination",
        type=str,
        help="The path of the destination in the bucket",
    )

    # Execute parse_args()
    args = parser.parse_args()

    upload_blob(args.Bucket, args.Source, args.Destination)


if __name__ == "__main__":
    main()
