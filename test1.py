import numpy
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

not_found = ["Acrab", "Alya", "Dabih", "Geminga", "Lich", "Marsic", "HAT-P-21", "Pipoltr"]

for star in not_found:
    print(f"Getting {star}...")
    image_url = f"https://alasky.u-strasbg.fr/hips-image-services/hips2fits?hips=CDS%2FP%2FDSS2%2Fcolor&width=500&height=500&fov=1&projection=SIN&coordsys=icrs&rotation_angle=0.0&object={star}&format=jpg"
    img_data = requests.get(image_url).content
    with open(f'/Users/jminding/Documents/Star Images/{star}.jpg', 'wb') as handler:
        handler.write(img_data)