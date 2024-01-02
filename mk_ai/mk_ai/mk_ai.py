from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import NoSuchElementException
import chromedriver_autoinstaller
import time

class MKAI:
    def __init__(self, headless=True):
        chromedriver_autoinstaller.install()
        options = Options()
        if headless:
            options.add_argument("--headless")
        self.driver = webdriver.Chrome(options=options)
    
    def classify(self, image_path):
        self.driver.get("https://starclassification.pages.dev/")
        self.driver.find_element(By.ID, "imageUpload").send_keys(image_path)
        i = 0
        while True:
            try:
                result = self.driver.find_element(By.ID, "result").text
                classification = result.split("\n")[0].split(": ")[1].strip()
                confidence = float(str(float(result.split("\n")[1].split(": ")[1].replace("%", "")) / 100)[:6])
                return classification, confidence
            except NoSuchElementException:
                if i == 0:
                    print("Awaiting classification...")
            i += 1
    

mkai = MKAI(headless=True)
print(mkai.classify("/Users/jminding/Documents/Research/Star Images 2/F/Achird.jpg"))

# requests.get("https://starclassification.pages.dev/?request=true&image=")