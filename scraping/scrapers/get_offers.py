"""
Offers Scraper Module

This module provides functionality to scrape offers related to car manufacturers.
It can download data from URLs, store the scraped data, and clear the stored data.

Classes:
    OfferScraper: Class for scraping offers related to manufacturer name.

Functions:
    get_header: Gets a list of column names from the given header file path.
    new_line: Generates a new line of batch data.
    download_url: Downloads offer data from a given URL.
    get_offers: Fetches a row of data for each offer link per manufacturer.
    save_offers: Stores scraped offers per manufacturer as a static CSV file.
    clear_list: Clears the stored manufacturer list.

"""

import os
import time
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha1

import pandas as pd
import requests
from bs4 import BeautifulSoup
from utils.logger import console_logger, file_logger

PATH_DATA = "data"
PATH_HEADER_FILE_PL = "header_pl.txt"
PATH_HEADER_FILE_EN = "header_en.txt"
MAX_THREADS = 8


class OfferScraper:
    """
    Scrapes offers related to manufacturer name
    Args:
        path_data_directory: path to a directory where data will be stored
        path_header_file_pl: path to file with features
    """

    def __init__(
        self,
        # path_data_directory=PATH_DATA,
        path_header_file_pl=PATH_HEADER_FILE_PL,
        path_header_file_en=PATH_HEADER_FILE_EN,
        max_threads=MAX_THREADS,
    ):
        self.path_data_directory = os.path.join(os.getcwd(), "data", "raw")
        self.path_header_file_pl = os.path.join(
            os.getcwd(), "data", "metadata", path_header_file_pl
        )
        self.path_header_file_en = os.path.join(
            os.getcwd(), "data", "metadata", path_header_file_en
        )
        self.max_threads = max_threads
        self.header_pl = self.get_header(self.path_header_file_pl)
        self.header_en = self.get_header(self.path_header_file_en)
        self.manufacturer = []

    def get_header(self, header_file_path) -> list:
        """
        Gets a list of column names from the given header file path
        :param header_file_path: path to the header file
        :return: a list of column names
        """
        with open(header_file_path, "r", encoding="utf-8") as file:
            header = [x.strip() for x in file.readlines()]

        return header

    def new_line(self, main_features: dict) -> dict:
        """
        Get a new line of a batch data
        :param main_features:   a dictionary of column names and according values
        :return:                a key, value dictionary
        """
        row = {column: main_features.get(column, None) for column in self.header_pl}

        return row

    def download_url(self, url_path: str) -> dict:
        """
        :param url_path:    url path to the offer per manufacturer
        :return:            a dictionary of offer's features
        """
        try:
            file_logger.info("Fetching %s", url_path)

            with requests.Session() as session:
                response = session.get(url_path)
                response.raise_for_status()

            soup = BeautifulSoup(response.text, features="lxml")

            params = soup.find_all(class_="offer-params__item")
            batch = {
                param.find("span", class_="offer-params__label")
                .text.strip(): param.find("div", class_="offer-params__value")
                .text.strip()
                for param in params
            }

            values = soup.find_all("li", class_="parameter-feature-item")
            batch.update({value.text.strip(): 1 for value in values})

            price_element = soup.find("span", class_="offer-price__number")
            price = price_element.text.strip() if price_element else ""
            batch["Cena"] = price

            currency_element = soup.find("span", class_="offer-price__currency")
            currency = currency_element.text.strip() if currency_element else ""
            batch["Waluta"] = currency

            price_details_element = soup.find("span", class_="offer-price__details")
            price_details = (
                price_details_element.text.strip() if price_details_element else ""
            )
            batch["Szczegóły ceny"] = price_details

            batch["url_path"] = url_path

            batch["id"] = sha1(url_path.lower().encode("utf-8")).hexdigest()

            batch = self.new_line(main_features=batch)

            batch["epoch"] = int(time.time())

            time.sleep(0.25)

            return batch

        except requests.RequestException as req_error:
            file_logger.error("Error %s while fetching %s", req_error, url_path)
            return {}
        except ValueError as value_error:
            file_logger.error(
                "ValueError %s while processing %s", value_error, url_path
            )
            return {}

    def get_offers(self, links: list) -> None:
        """
        Gets a row of data for each offer link per manufacturer
        :param links: a list of links to the offers
        :return: None
        """
        max_workers = max(self.max_threads, 1)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            rows = executor.map(self.download_url, links)
            self.manufacturer.extend(row for row in rows if row is not None)

    def save_offers(self, manufacturer: str) -> None:
        """
        Stores scraped offers per manufacturer as a static file
        :param manufacturer:    car manufacturer name
        :return:                None
        """
        file_logger.info("Saving %s offers", manufacturer)
        file_logger.info("Found %s offers", len(self.manufacturer))
        console_logger.info("Found %s offers", len(self.manufacturer))

        file_path = os.path.join(
            self.path_data_directory, f"{manufacturer.strip()}.csv"
        )

        if os.path.isfile(file_path):
            # Load existing CSV file
            existing_data_frame = pd.read_csv(file_path)

            # Filter out duplicates based on 'id' column
            existing_ids = existing_data_frame["id"].tolist()
            new_rows = [
                row for row in self.manufacturer if row["id"] not in existing_ids
            ]
            new_data_frame = pd.DataFrame(new_rows)

            # Append new rows to existing data
            data_frame = pd.concat(
                [existing_data_frame, new_data_frame], ignore_index=True
            )
        else:
            # Create new DataFrame if the file doesn't exist
            data_frame = pd.DataFrame(self.manufacturer)

        # Drop duplicates based on 'id' column
        data_frame.drop_duplicates(subset="id", inplace=True)

        # Save the DataFrame to a CSV file
        data_frame.to_csv(file_path, index=False)

        file_logger.info("Saved %s offers", manufacturer)

    def clear_list(self) -> None:
        """
        Clears the manufacturer list
        :return: None
        """
        self.manufacturer = []
