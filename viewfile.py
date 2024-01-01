from astropy.io import fits
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import math

# star = "Adhil"
#
# hdu = fits.open(f"Star Images/{star}_masked.fits")
# data = hdu[0].data
# plt.imshow(data)
# plt.show()
#
# # Remove all nan values from data
# data = np.nan_to_num(data)
# # data = list([list(i) for i in data])
# # data = np.array(data)
# print(np.average(data))

df = pd.read_csv("star_data.csv")
# print(df[df["Star Name"] == star]["L/L_o"])
# print(df[df["Star Name"] == star]["Distance (pc)"])

brightness = []
luminosity = []

for star in list(df["Star Name"]):
    image = Image.open(f"Star Images/{star}_removedbg.jpg")
    # data = hdu[0].data
    # Remove all nan values from data
    data = list(image.getdata())
    data = [i for i in data if i != (0, 0, 0)]
    # print(data)
    r = 0
    g = 0
    b = 0
    for i in range(len(data)):
        r += data[i][0]
        g += data[i][1]
        b += data[i][2]
    r = r / len(data)
    g = g / len(data)
    b = b / len(data)
    # perceived_brightness = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    perceived_brightness = math.sqrt(0.299 * (r ** 2) + 0.587* (g ** 2) + 0.114 * (b ** 2))
    # perceived_brightness = np.median([r, g, b])
    # print(perceived_brightness)
    # exit()
    # data = np.nan_to_num(data)
    brightness.append(perceived_brightness)
    luminosity.append(abs(df[df["Star Name"] == star]["Distance (pc)"].values[0] * df[df["Star Name"] == star]["L/L_o"].values[0]))
    # print(np.average(data))
    # print(df[df["Star Name"] == star]["L/L_o"].values[0])

plt.plot(brightness, luminosity, "o")
# plt.ylim(0, 500)
plt.show()

print(list(np.array(brightness)))