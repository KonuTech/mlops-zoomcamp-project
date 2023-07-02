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
        path_data_directory=PATH_DATA,
        path_manufacturers_file=PATH_MANUFACTURERS_FILE,
    ):
        self.path_manufacturers_file = os.path.join(
            os.getcwd(), PATH_DATA, "inputs", path_manufacturers_file
        )
        self.path_data_directory = os.path.join(os.getcwd(), PATH_DATA, "inputs")
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
        console_logger.info(f"Scraping page: {i}")
        file_logger.info(f"Scraping page: {i}")

        with requests.Session() as session:
            response = session.get(f"{path}?page={i}")
            response.raise_for_status()

        soup = BeautifulSoup(response.text, features="lxml")

        car_links_section = soup.find("main", attrs={"data-testid": "search-results"})

        if car_links_section is None:
            console_logger.warning(f"No car links found on page {i}")
            file_logger.warning(f"No car links found on page {i}")
            return []

        links = [
            x.find("a", href=True)["href"]
            for x in car_links_section.find_all("article")
        ]

        console_logger.info(f"Found {len(links)} links")
        file_logger.info(f"Found {len(links)} links")

        return links

    def scrap_manufacturer(self, manufacturer: str) -> None:
        """
        Scrapes manufacturer data from otomoto.pl
        :param manufacturer:    car manufacturer name
        :return:                None
        """
        manufacturer = manufacturer.strip()

        console_logger.info(f"Start of scraping the manufacturer: {manufacturer}")
        file_logger.info(f"Start of scraping the manufacturer: {manufacturer}")

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

        except Exception as e:
            file_logger.error(f"Error {e} while searching for last_page_num")
            last_page_num = 1

        last_page_num = min(last_page_num, 1000)

        console_logger.info(f"Manufacturer has: {last_page_num} subpages")
        file_logger.info(f"Manufacturer has: {last_page_num} subpages")

        pages = range(1, last_page_num + 1)

        for p, page in enumerate(pages):
            links = self.get_links(path=url, i=page)
            self.offers.get_offers(links=links)

            time.sleep(0.2)

        # Save the list of offers
        self.offers.save_offers(manufacturer=manufacturer)

        console_logger.info(f"End of scraping the manufacturer: {manufacturer}")
        file_logger.info(f"End of scraping the manufacturer: {manufacturer}")

    def scrap_all_manufacturers(self) -> None:
        """
        Loops over the list of manufacturer names to scrape data for each one of them
        :return: None
        """
        console_logger.info("Starting scraping cars...")
        file_logger.info("Starting scraping cars...")

        for m, manufacturer in enumerate(self.manufacturers):
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

        combined_data = []

        for f, filename in enumerate(filenames):
            try:
                data = pd.read_csv(filename)
                data.columns = self.offers.header_en
                combined_data.append(data)

            except Exception as e:
                file_logger.error(f"Error {e} while searching for {filename}")

        df = pd.concat(combined_data, ignore_index=True)
        df.to_csv(
            os.path.join(self.path_data_directory, OUTPUT_NAME),
            index=False,
            encoding="utf-8",
        )

        console_logger.info(f"Appended data saved as {OUTPUT_NAME}")
        file_logger.info(f"Appended data saved as {OUTPUT_NAME}")
