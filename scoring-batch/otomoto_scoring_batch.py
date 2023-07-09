import json
import sys

import pandas as pd
import requests

# Add the desired path to sys.path
sys.path.append("/home/konradballegro/scraping")
from scrapers.offers_scraper import ManufacturerScraper

CONFIG_PATH = "/home/konradballegro/scoring-batch/config/config.json"


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


def data_read(file_path: str) -> str:
    """
    Reads the data from the specified file.
    Args:
        file_path (str): Path to the file to be read.
    Returns:
        data_json (str): JSON string containing the read data.
    """
    # logging.info(f"Reading data from file: {file_path}")
    df = pd.read_csv(file_path, index_col=False)
    data_json = df.to_json()
    return data_json


url = "http://127.0.0.1:9696/predict"
try:
    print("GOT A JSON FILE")
    response = requests.post(url, json=data_read(file_path=FILE_PATH))
    print("GOT A RESPONSE", response)
    response.raise_for_status()
    result = response.json()
    print(result)
except requests.exceptions.RequestException as e:
    print(f"Request failed: {e}")
