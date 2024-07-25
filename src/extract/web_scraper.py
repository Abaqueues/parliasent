import re
import requests
import os
from bs4 import BeautifulSoup as bs

# Use requests library to access the website
_URL = "https://www.theyworkforyou.com/pwdata/scrapedxml/debates/"
_PAGE = requests.get(_URL)

# Define scraper parameter and output folder path
_YEAR = 2013
_XML_OUTPUT_FOLDER = ".\\TWFYScraper\\xml\\"

# Check whether the output folder exists
os.makedirs(_XML_OUTPUT_FOLDER, exist_ok=True)

print("Scraping TWFY .xml debate transcripts from " + f"{_YEAR} onwards...")
print("Outputting to " + f"'{_XML_OUTPUT_FOLDER}'..." + "\n")

# Function to check whether .xml file corresponds to debates after given date
def isAfter(string):
    number = re.findall(r'\d+', string)
    if int(number[0]) >= _YEAR:
        return True
    else:
        return False

# Instantiate BeautifulSoup object
soup = bs(_PAGE.content, "html.parser")

urls = []
names = []

# Loops through all classes tagged "a" and adds relevant names/URLs to list
for i, link in enumerate(soup.findAll("a")):
    _FULLURL = _URL + link.get("href")
    if _FULLURL.endswith(".xml"):
        if isAfter(_FULLURL):
            print(_FULLURL)
            urls.append(_FULLURL)
            names.append(soup.select("a")[i].attrs["href"])

# Combines the names/URLs lists 
names_urls = zip(names, urls)
print(names_urls)

# Function to download .xml files 
def scrapeFile(name, url):
    print(_XML_OUTPUT_FOLDER + name)
    if os.path.exists(_XML_OUTPUT_FOLDER + name) == True:
        print(f"File '{name}' already exists. Skipping..." + "\n")
    else:
        print("Downloading %s" % url + ".\n")
        r = requests.get(url)
        with open(_XML_OUTPUT_FOLDER + name.split("\\")[-1], "wb") as f:
            f.write(r.content)
        
for name, url in zip(names, urls):
    print(f"{name}" + f"{url}")
            
# Loops through and downloads the files using the URLs in the zipped list
print("Initiating scraping loop")
for name, url in list(names_urls):
    print(f"Scraping '{name}'.")
    scrapeFile(name, url)
    
print("Scraping complete...\n")
    #   + f"{skipped}" + " files skipped...\n"
    #   + f"{downloaded}" + " files downloaded...\n")