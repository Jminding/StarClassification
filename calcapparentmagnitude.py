import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

def calc_luminosity(img_path):
    im = cv2.imread(img_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    pixel_sum_value = np.sum(im)
    avg_img_value = pixel_sum_value / (im.shape[0] * im.shape[1])
    bw = 50
    mask = np.ones(im.shape[:2], dtype = "uint8")
    cv2.rectangle(mask, (bw,bw),(im.shape[1]-bw,im.shape[0]-bw), 0, -1)
    output = cv2.bitwise_and(im, im, mask = mask)
    pixel_sum_value_2 = np.sum(output)
    avg_bg_value = pixel_sum_value_2 / 90000
    luminosity = pixel_sum_value - avg_bg_value * 500 * 500
    return luminosity

def calc_apparent_magnitude(img_path):
    luminosity = calc_luminosity(img_path)
    vega_luminosity = calc_luminosity("Star Images/Vega.jpg")
    apparent_magnitude = -2.5 * np.log10(luminosity / vega_luminosity)
    return apparent_magnitude