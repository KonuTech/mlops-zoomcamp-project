"""
Manufacturer Scraper Module

This module provides functionality to scrape data related to car offers from www.otomoto.pl.
It scrapes data for each manufacturer listed in a file and saves the scraped data into CSV files.

Classes:
    ManufacturerScraper: Class for scraping data related to offers of cars from www.otomoto.pl.

Functions:
    get_manufacturers: Gets a list of manufacturers from a static file.
    get_links: Gets links of car offers from a web page.
    scrap_manufacturer: Scrapes manufacturer data from otomoto.pl.
    scrap_all_manufacturers: Loops over the list of manufacturer names
    to scrape data for each one of them.
    dump_data: Appends offers data and stores it as a static CSV file.

"""

import os
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup
from scrapers.get_offers import OfferScraper
from utils.logger import console_logger, file_logger

PATH_DATA = "data"
PATH_MANUFACTURERS_FILE = "manufacturers.txt"
URL_BASE = "https://www.otomoto.pl/osobowe/"
OUTPUT_NAME = "offers.csv"


class ManufacturerScraper:
    """
    Scrapes data related to offers of cars from www.otomoto.pl
    Args:
        path_data_directory: path to a directory where data will be stored
        path_manufacturers_file: path to a file with names of manufacturers
    """

    console_logger.info("Initializing a scraper")
    file_logger.info("Initializing a scraper")

    def __init__(
        self,
        # path_data_directory,
        path_manufacturers_file=PATH_MANUFACTURERS_FILE,
    ):
        self.path_manufacturers_file = os.path.join(
            os.getcwd(), PATH_DATA, "metadata", path_manufacturers_file
        )
        self.path_data_directory = os.path.join(os.getcwd(), PATH_DATA, "raw")
        self.manufacturers = self.get_manufacturers()
        self.offers = OfferScraper()

    def get_manufacturers(self) -> list:
        """
        Gets a list of manufacturers from a static file
        :return: a list of car manufacturers' names
        """
        with open(self.path_manufacturers_file, "r", encoding="utf-8") as file:
            manufacturers = [line.strip() for line in file]

        return manufacturers

    @staticmethod
    def get_links(path: str, i: str) -> list:
        """
        Gets links of car offers from a web page
        :param path:    path to a web page
        :param i:       web page number
        :return:        a list of links
        """
        console_logger.info("Scraping page: %s", i)
        file_logger.info("Scraping page: %s", i)

        with requests.Session() as session:
            response = session.get(f"{path}?page={i}")
            response.raise_for_status()

        soup = BeautifulSoup(response.text, features="lxml")

        car_links_section = soup.find("main", attrs={"data-testid": "search-results"})

        if car_links_section is None:
            console_logger.warning("No car links found on page %s", i)
            file_logger.warning("No car links found on page %s", i)
            return []

        links = [
            x.find("a", href=True)["href"]
            for x in car_links_section.find_all("article")
        ]

        console_logger.info("Found %s links", len(links))
        file_logger.info("Found %s links", len(links))

        return links

    def scrap_manufacturer(self, manufacturer: str) -> None:
        """
        Scrapes manufacturer data from otomoto.pl
        :param manufacturer:    car manufacturer name
        :return:                None
        """
        manufacturer = manufacturer.strip()

        console_logger.info("Start of scraping the manufacturer: %s", manufacturer)
        file_logger.info("Start of scraping the manufacturer: %s", manufacturer)

        # Clear the list of offers
        self.offers.clear_list()

        url = f"{URL_BASE}{manufacturer}"

        try:
            with requests.Session() as session:
                response = session.get(url)
                response.raise_for_status()

            soup = BeautifulSoup(response.text, features="lxml")
            last_page_num = int(
                soup.find_all("li", attrs={"data-testid": "pagination-list-item"})[
                    -1
                ].text
            )

        except requests.exceptions.RequestException as request_exc:
            file_logger.error("Error during HTTP request: %s", request_exc)
            last_page_num = 1

        except Exception as other_exc:
            file_logger.error("Unexpected error: %s", other_exc)
            last_page_num = 1

        last_page_num = min(last_page_num, 1000)

        console_logger.info("Manufacturer has: %s subpages", last_page_num)
        file_logger.info("Manufacturer has: %s subpages", last_page_num)

        for page in range(1, last_page_num + 1):
            links = self.get_links(path=url, i=page)
            self.offers.get_offers(links=links)

            time.sleep(0.2)

        # Save the list of offers
        self.offers.save_offers(manufacturer=manufacturer)

        console_logger.info("End of scraping the manufacturer: %s", manufacturer)
        file_logger.info("End of scraping the manufacturer: %s", manufacturer)

    def scrap_all_manufacturers(self) -> None:
        """
        Loops over the list of manufacturer names to scrape data for each one of them
        :return: None
        """
        console_logger.info("Starting scraping cars...")
        file_logger.info("Starting scraping cars...")

        for manufacturer in self.manufacturers:
            csv_file_path = os.path.join("data", "raw", f"{manufacturer}.csv")
            if os.path.isfile(csv_file_path):
                console_logger.info(
                    "Skipping scraping for %s. CSV exists.", manufacturer
                )
                file_logger.info("Skipping scraping for %s. CSV exists.", manufacturer)
            else:
                self.scrap_manufacturer(manufacturer=manufacturer)

        console_logger.info("End of scraping manufacturers")
        file_logger.info("End of scraping manufacturers")

    def dump_data(self) -> None:
        """
        Appends offers data and stores it as a static file
        :return: None
        """
        console_logger.info("Appending the data...")
        file_logger.info("Appending the data...")

        filenames = [
            os.path.join(self.path_data_directory, f"{manufacturer.strip()}.csv")
            for manufacturer in self.manufacturers
        ]

        output_file_path = os.path.join(self.path_data_directory, OUTPUT_NAME)

        if os.path.isfile(output_file_path):
            previous_data = pd.read_csv(output_file_path)
        else:
            previous_data = pd.DataFrame()

        for filename in filenames:
            try:
                data = pd.read_csv(filename)
                data.columns = self.offers.header_en

                # Append unique rows to the output file
                unique_rows = data[~data["ID"].isin(previous_data["ID"])]
                unique_rows.to_csv(
                    output_file_path,
                    mode="a",
                    index=False,
                    header=not os.path.isfile(output_file_path),
                    encoding="utf-8",
                )
                previous_data = previous_data.append(unique_rows, ignore_index=True)

            except pd.errors.EmptyDataError as empty_data_exc:
                file_logger.error("Empty data error: %s", empty_data_exc)

            except pd.errors.ParserError as parser_exc:
                file_logger.error("Parser error: %s", parser_exc)

            except Exception as other_exc:
                file_logger.error("Unexpected error: %s", other_exc)

        console_logger.info("Appended data saved as %s", OUTPUT_NAME)
        file_logger.info("Appended data saved as %s", OUTPUT_NAME)
