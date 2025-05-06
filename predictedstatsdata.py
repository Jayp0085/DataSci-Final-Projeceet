from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import pandas as pd

options = webdriver.ChromeOptions()
driver = webdriver.Chrome(options=options)
driver.get("https://fantasy.espn.com/basketball/players/projections")
button_xpath = '//*[@id="fitt-analytics"]/div/div[5]/div[2]/div[2]/div[1]/div/div[2]/div[2]/div/button[1]'
WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, button_xpath))).click()
WebDriverWait(driver, 10).until(
    EC.presence_of_all_elements_located((By.TAG_NAME, "table"))
)
soup = BeautifulSoup(driver.page_source, 'html.parser')
tables = soup.find_all('table')
dfs = pd.read_html(str(tables))

if len(dfs) >= 2:
    df_combined = pd.concat([dfs[0], dfs[1]], axis=1)
driver.quit()
df_combined.to_csv("espn_player_projections.csv",index=False)
