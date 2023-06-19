import os
import time
from hashlib import sha1
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests
from bs4 import BeautifulSoup

from utils.logger import console_logger
from utils.logger import file_logger


PATH_DATA = 'data'
PATH_HEADER_FILE = 'header.txt'
MAX_THREADS = 8


class OfferScraper:
    """
    Scraps offers related to manufacturer name
    Args:
        path_data_directory: path to a directory where data will be stored
        path_header_file: path to file with features
    """

    def __init__(self, path_data_directory=PATH_DATA, path_header_file=PATH_HEADER_FILE, max_threads=MAX_THREADS):
        self.path_header_file = os.path.join(os.getcwd(), 'inputs', path_header_file)
        self.path_data_directory = os.path.join(os.getcwd(), 'outputs', path_data_directory)
        self.max_threads = max_threads
        self.header = self.get_header()
        self.manufacturer = []

    def get_header(self) -> list:
        """
        Gets a list of column names
        :return: a list of column names
        """
        with open(self.path_header_file, 'r', encoding='utf-8') as file:
            header = [x.strip() for x in file.readlines()]

        return header

    def new_line(self, main_features: dict) -> dict:
        """
        Get a new line of a batch data
        :param main_features:   a dictionary of column names and according values
        :return:                a key, value dictionary
        """
        row = {column: main_features.get(column, None) for column in self.header}

        return row

    def download_url(self, url_path: str) -> dict:
        """
        :param url_path:    url path to the offer per manufacturer
        :return:            a dictionary of offer's features
        """
        try:
            file_logger.info(f'Fetching {url_path}')

            with requests.Session() as session:
                response = session.get(url_path)
                response.raise_for_status()

            soup = BeautifulSoup(response.text, features='lxml')

            params = soup.find_all(class_='offer-params__item')
            batch = {
                param.find('span', class_='offer-params__label').text.strip():
                    param.find('div', class_='offer-params__value').text.strip() for param in params
            }

            values = soup.find_all('li', class_='parameter-feature-item')
            batch.update({value.text.strip(): 1 for value in values})

            price = ''.join(soup.find('span', class_='offer-price__number').text.strip().split()[:-1])
            batch['Cena'] = price

            currency = soup.find('span', class_='offer-price__currency').text.strip()
            batch['Waluta'] = currency

            price_details = soup.find('span', class_='offer-price__details').text.strip()
            batch['Szczegóły ceny'] = price_details

            batch['url_path'] = url_path

            batch['id'] = sha1(url_path.lower().encode('utf-8')).hexdigest()

            batch = self.new_line(main_features=batch)

            time.sleep(0.25)

            return batch

        except Exception as e:
            file_logger.error(f'Error {e} while fetching {url_path}')

    def get_offers(self, links: list) -> None:
        """
        Gets a row of data for each offer link per manufacturer
        :param links: a link to the offer
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
        file_logger.info(f'Saving {manufacturer} offers')
        file_logger.info(f'Found {len(self.manufacturer)} offers')
        console_logger.info(f'Found {len(self.manufacturer)} offers')

        df = pd.DataFrame(self.manufacturer)
        df.to_csv(os.path.join(self.path_data_directory, f'{manufacturer.strip()}.csv'), index=False)

        file_logger.info(f'Saved {manufacturer} offers')

    def clear_list(self) -> None:
        """
        Clears manufacturer list
        :return: None
        """
        self.manufacturer = []
