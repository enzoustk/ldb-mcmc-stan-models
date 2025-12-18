from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd

def extract_match_links(mainpage='https://lnb.com.br/ldb/tabela-de-jogos'):
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome(options=options)
    driver.get(mainpage)
    WebDriverWait(driver, 15).until(
        EC.presence_of_all_elements_located((By.CSS_SELECTOR, "a"))
    )
    soup = BeautifulSoup(driver.page_source, "html.parser")
    driver.quit()

    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "/partidas/" in href:
            full_url = href if href.startswith("http") else f"https://lnb.com.br{href}"
            links.append(full_url)

    return list(dict.fromkeys(links))  # remove duplicados mantendo ordem
