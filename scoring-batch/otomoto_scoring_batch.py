import json
import sys

import pandas as pd
import requests

# Add the desired path to sys.path
sys.path.append("/home/konradballegro/scraping")
from scrapers.offers_scraper import ManufacturerScraper

CONFIG_PATH = "/home/konradballegro/scoring-batch/config/config.json"
URL = "http://127.0.0.1:9696/predict"


def data_read(file_path: str) -> str:
    """
    Reads the data from the specified file.
    Args:
        file_path (str): Path to the file to be read.
    Returns:
        data_json (str): JSON string containing the read data.
    """
    df = pd.read_csv(file_path, index_col=False)
    data_json = df.to_json()
    return data_json

def main():
    """
    Main function to execute the script.

    Reads the configuration file, performs web scraping, and makes a POST request
    to a specified URL with the data obtained from the file.

    Raises:
    requests.exceptions.RequestException: If the request to the URL fails.
    """
    with open(CONFIG_PATH, encoding="UTF-8") as json_file:
        config = json.load(json_file)


    FILE_PATH = config["FILE_PATH"]
    PATH_DATA = "data"
    PATH_MANUFACTURERS_FILE = "manufacturers_batch.txt"

    scraper = ManufacturerScraper(
        path_manufacturers_file=PATH_MANUFACTURERS_FILE, path_data_directory=PATH_DATA
    )
    scraper.scrap_all_manufacturers()
    scraper.dump_data()

    try:
        response = requests.post(URL, json=data_read(file_path=FILE_PATH))
        response.raise_for_status()
        result = response.json()
        print(result)
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    main()
