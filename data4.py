from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import pandas as pd
import time

options = Options()
options.headless = True
driver = webdriver.Chrome(options=options)
url = "https://www.sportscardspro.com/console/basketball-cards-2024-panini-nba-hoops"
driver.get(url)
try:
    WebDriverWait(driver, 20).until(
        EC.presence_of_element_located((By.ID, "games_table"))
    )
except:
    print("Table did not load in time.")
    driver.quit()
    exit()
last_height = driver.execute_script("return document.body.scrollHeight")
while True:
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(2)  # Wait for new data to load
    new_height = driver.execute_script("return document.body.scrollHeight")
    if new_height == last_height:
        break
    last_height = new_height
soup = BeautifulSoup(driver.page_source, 'html.parser')
table = soup.find('table', id='games_table')
headers = [th.text.strip() for th in table.find('thead').find_all('th')]
rows = []
for tr in table.find('tbody').find_all('tr'):
    cells = [td.text.strip() for td in tr.find_all('td')]
    if cells:
        rows.append(cells)
df = pd.DataFrame(rows, columns=headers)
df.to_csv('2024_panini_nba_hoops_prices.csv', index=False)
driver.quit()

df = pd.read_csv('2024_panini_nba_hoops_prices.csv')
df.drop(columns=['Unnamed: 4'], inplace=True)
df.to_csv('2024_panini_nba_hoops_prices.csv', index=False)

