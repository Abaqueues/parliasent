from src.extract.web_scraper import *

def main():
    scrape_data()

def scrape_data():
    scraper = WebScraper()
    scraper.compile_names_urls()

if __name__ == "__main__":
    main()