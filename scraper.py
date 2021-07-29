import os
import csv
import time
from tqdm.auto import tqdm
import bs4
import urllib
from typing import List
import concurrent.futures
from urllib.request import Request
from datetime import date, datetime, timedelta

# globals
DATA_DIR = os.path.join(os.getcwd(), 'data')
MAIN_URL = "https://www.horoscope.com/us/horoscopes/"
CATEGORIES = ['general', 'love', 'career', 'wellness', 'birthday']
MAX_THREADS = 50
SIGNS = {
    'aries': 1,
    'taurus': 2,
    'gemini': 3,
    'cancer': 4,
    'leo': 5,
    'virgo': 6,
    'libra': 7,
    'scorpio': 8,
    'sagittarius': 9,
    'capricorn': 10,
    'aquarius': 11,
    'pisces': 12
}

# scrape data from these dates (June 16th, 2020 -> June 16th, 2021)

# generate url function
def generate_url(category:str, sign:str, dates:List):
    """Generate URL strings based on parameters.
    Args:
    -----
    category: horoscope category
    sign: horscope sign
    date: list of all dates to be scraped 
    --------
    Returns:
    --------
    final_url: Final URL string that can be passed to urllib function.
    """
    final_urls = []
    if category == 'birthday':
        cat = 'general'
        end_part = 'general-birthday.aspx?'
        sign_type = ''
    else:
        cat = category
        end_part = 'archive.aspx?'
        sign_type = f"sign={str(SIGNS[sign])}"

    for d in dates:
        final_urls.append(f"{MAIN_URL + cat}/horoscope-{end_part + sign_type}&laDate={d}")

    return final_urls

def urllib_request(url, try_count=0):
    """Makes urllib requests with try and catch, if it throws a ConnectionResetError.
    If error thrown, it will try 3 more times every minute before it exits the function.
    Args:
    --------
    url: Site URL
    --------
    Returns:
    html: Request object
    """
    try:
        html = urllib.request.urlopen(Request(url, headers={'User-Agent': 'Mozilla'}))
        time.sleep(0.25)
        return html
    except urllib.error.URLError or ConnectionResetError:
        print(f"Trying again... \n{try_count+1} times.")
        time.sleep(60)
        try_count += 1
        if try_count == 3:
          print("Ended after many tries.")
          exit
        else:
          urllib_request(url, try_count=try_count)
        
def scrape_urls(url):
    """Get scraped texts from URLs."""
    html = urllib_request(url).read()
    raw = bs4.BeautifulSoup(html, 'html.parser')
    if 'general-birthday.aspx' in url:
        results = raw.find('div', {'class': 'grid-single-m'}).find('p').text.strip()
        return results
    else:
        results = raw.find('div', {'class': 'main-horoscope'}).find('p')
        return str(results).split('</strong> - ')[-1].split('</p>')[0]
    return 'none found'

def get_texts(res):
    yield from res

def write_to_file(sign, category, dates, texts):
    data_file_path = os.path.join(DATA_DIR, f'horoscope_final.csv')
    
    for d, t in zip(dates, texts):
        line = [str(sign), str(category), str(d), str(t)]
        with open(data_file_path, 'a', encoding='utf-8') as data_file:
            writer = csv.writer(data_file, delimiter=',')
            writer.writerow(line)

def scraper(start_date:str, end_date:str):
    """This is the main function of this script that scrapes all the data and saves it to a .csv file."""
    sdate = datetime.strptime(start_date, '%Y%m%d').date()
    edate = datetime.strptime(end_date, '%Y%m%d').date()
    delta = edate - sdate
    dates_to_scrape = [(sdate+timedelta(days=i)).strftime('%Y%m%d') for i in range(delta.days + 1)]
    
    for sign in SIGNS.keys():
        print(f"-----\t{sign}\t-----")
        for cat in tqdm(CATEGORIES):
            all_urls = generate_url(cat, sign, dates_to_scrape)
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
                res = executor.map(scrape_urls, all_urls)

            write_to_file(sign, cat, dates_to_scrape, list(get_texts(res)))
        time.sleep(5)
        
if __name__ == "__main__":
    
    scraper(start_date='20200616', end_date='20210616')
    print("Complete!")