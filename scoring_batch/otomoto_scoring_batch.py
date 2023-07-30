"""
This script reads the data from a specified file,
performs web scraping,
and makes a POST request to a specified URL
with the data obtained from the file.
"""

import json
import sys

import pandas as pd
import requests

# Add the desired path to sys.path
sys.path.append("/home/konradballegro/scraping")
from scrapers.offers_scraper import ManufacturerScraper

CONFIG_PATH = "/home/konradballegro/scoring_batch/config/config.json"
with open(CONFIG_PATH, encoding="UTF-8") as json_file:
    config = json.load(json_file)


FILE_PATH = config["FILE_PATH"]
PATH_DATA = "data"
PATH_MANUFACTURERS_FILE = "manufacturers_batch.txt"
URL = "http://127.0.0.1:9696/predict"


def data_read(file_path: str) -> str:
    """
    Reads the data from the specified file.
    Args:
        file_path (str): Path to the file to be read.
    Returns:
        data_json (str): JSON string containing the read data.
    """
    data_frame = pd.read_csv(file_path, index_col=False, low_memory=False)
    data_json = data_frame.to_json()
    return data_json


def main():
    """
    Main function to execute the script.

    Reads the configuration file, performs web scraping, and makes a POST request
    to a specified URL with the data obtained from the file.

    Raises:
    requests.exceptions.RequestException: If the request to the URL fails.
    """

    scraper = ManufacturerScraper(path_manufacturers_file=PATH_MANUFACTURERS_FILE)
    scraper.scrap_all_manufacturers()
    scraper.dump_data()

    try:
        response = requests.post(URL, json=data_read(file_path=FILE_PATH), timeout=10)
        response.raise_for_status()
        result = response.json()
        print(result)
    except requests.exceptions.RequestException as ex:
        print(f"Request failed: {ex}")


if __name__ == "__main__":
    main()
