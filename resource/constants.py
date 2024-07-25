import requests
from pathlib import Path

#  web_scraper.py

URL = "https://www.theyworkforyou.com/pwdata/scrapedxml/debates/"
PAGE = requests.get(URL)

YEAR = [2013]
XML_OUTPUT_FOLDER = r"data/xml"

