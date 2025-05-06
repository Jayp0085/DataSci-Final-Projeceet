from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import pandas as pd
import time

options = webdriver.ChromeOptions()
options.add_argument('--headless')  # not opening google broswer
driver = webdriver.Chrome(options=options)

driver.get("https://fantasy.espn.com/basketball/players/projections")

proj_button_xpath = '//*[@id="fitt-analytics"]/div/div[5]/div[2]/div[2]/div[1]/div/div[2]/div[2]/div/button[1]'
WebDriverWait(driver, 10).until(
    EC.element_to_be_clickable((By.XPATH, proj_button_xpath))
).click()

all_data = []

page = 1
while True:
    print(page) #should get to 22
    WebDriverWait(driver, 10).until(
        EC.presence_of_all_elements_located((By.TAG_NAME, "table"))
    )

    soup = BeautifulSoup(driver.page_source, 'html.parser')
    tables = soup.find_all('table')
    
    if len(tables) >= 2:
        dfs = pd.read_html(str(tables))
        df_combined = pd.concat([dfs[0], dfs[1]], axis=1)
        all_data.append(df_combined)
    else:
        break
    #clicking arrow to go next page and then scraping data there
    try:
        next_button_xpath = '//*[@id="fitt-analytics"]/div/div[5]/div[2]/div[3]/div/div/div/div/nav/button[2]'
        next_button = WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable((By.XPATH, next_button_xpath))
        )
        #make sure next button is there
        if "Button__disabled" in next_button.get_attribute("class"):
            break
        #clicks button and sleep 2 to allow page to load
        next_button.click()
        page += 1
        time.sleep(2)
    except:
        break


driver.quit()
final_df = pd.concat(all_data, ignore_index=True)
final_df.to_csv("espn_fantasy_all_projections.csv", index=False)
