from modules.scrapers.offers_scraper import ManufacturerScraper


if __name__ == '__main__':
    manufacturer_scraper = ManufacturerScraper()
    manufacturer_scraper.scrap_all_manufacturers()
    manufacturer_scraper.dump_data()
