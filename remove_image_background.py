import numpy as np
import pandas
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import NoSuchElementException
import os
os.environ["LANG"] = "en_US.UTF-8"
import chromedriver_autoinstaller
import matplotlib.pyplot as plt
import bs4
import requests
from unidecode import unidecode
import re
import pandas as pd
from sklearn.metrics import r2_score

chromedriver_autoinstaller.install()

for file in sorted(os.listdir("Star Images")):
    if file.endswith(".jpg"):
        print("Doing " + file)
        chrome_options = Options()
        # chrome_options.add_argument("--headless")
        chrome_options.add_argument("download.default_directory=" + os.path.join(os.getcwd(), "Star Images Removed Backgrounds"))
        driver = webdriver.Chrome(options=chrome_options)
        driver.get("https://remove.bg")
        driver.find_element(By.CLASS_NAME, '!border !border-transparent rounded-full font-bold transition ease-in-out text-center font-body no-underline hover:no-underline inline-flex items-center justify-center text-2xl px-8 py-2.5 text-white !bg-primary hover:!bg-primary-hover active:!bg-primary-hover active:scale-[0.98] focus:outline-none focus-visible:outline-none focus:ring-none focus-visible:ring focus-visible:ring-offset-2 focus-visible:ring-primary-hover').click()
        driver.send_keys(os.path.join(os.getcwd(), "Star Images", file))
        driver.find_element(By.CLASS_NAME, 'btn btn-primary btn-download mr-2 mr-md-0').click()
        driver.quit()