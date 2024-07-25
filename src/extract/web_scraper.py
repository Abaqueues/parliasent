import re
import requests
import os
from resource.constants import *
from bs4 import BeautifulSoup as bs

class WebScraper:

    def __init__(self):
        pass

    def check_date_is_after(string):
        number = re.findall(r'\d+', string)
        if int(number[0]) >= YEAR:
            return True
        else:
            return False

    def compile_names_urls(self):
        soup = bs(PAGE.content, "html.parser")

        urls = []
        names = []

        for i, link in enumerate(soup.findAll("a")):
            FULL_URL = URL + link.get("href")
            if FULL_URL.endswith(".xml"):
                if self.check_date_is_after(FULL_URL):
                    print(FULL_URL)
                    urls.append(FULL_URL)
                    names.append(soup.select("a")[i].attrs["href"])

        for name, url in zip(names, urls):
            print(f"{name}" + f"{url}")

        print("Initiating scraping loop")
        for name, url in list(names_urls):
            print(f"Scraping '{name}'.")
            self.scrape_file(name, url)
        print("Scraping complete...\n")

    def scrape_file(self, name, url):
        print(XML_OUTPUT_FOLDER + name)
        if os.path.exists(XML_OUTPUT_FOLDER + name) == True:
            print(f"File '{name}' already exists. Skipping..." + "\n")
        else:
            print("Downloading %s" % url + ".\n")
            r = requests.get(url)
            with open(XML_OUTPUT_FOLDER + name.split("\\")[-1], "wb") as f:
                f.write(r.content)