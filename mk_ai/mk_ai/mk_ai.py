import requests
import js2py
import cv2

class MKAI:
    def __init__(self, image_path):
        self.image_path = image_path

    def load_image(self):
        self.image = cv2.imread(self.image_path)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.image = cv2.resize(self.image, (224, 224))
        

requests.get("https://starclassification.pages.dev/?request=true&image=")