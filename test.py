import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from PIL import Image
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import r2_score
import pandas as pd

test = Image.open("Star Images/Arcturus_removedbg.jpg")
test_array = list(image.img_to_array(test))
for i in range(len(test_array)):
    test_array[i] = list(test_array[i])
    for j in range(len(test_array[i])):
        test_array[i][j] = list(test_array[i][j])

new_array = []
for i in range(len(test_array)):
    for j in range(len(test_array[i])):
        new_array.append(test_array[i][j])

# remove all [0.0, 0.0, 0.0] from new_array
new_array = [i for i in new_array if i != [0.0, 0.0, 0.0]]

# find average of r,g,b values in new_array
r = 0
g = 0
b = 0
for i in range(len(new_array)):
    r += new_array[i][0]
    g += new_array[i][1]
    b += new_array[i][2]
r = r / len(new_array)
g = g / len(new_array)
b = b / len(new_array)
avg_val = [r / 255, g / 255, b / 255]
print(avg_val)

# display avg_val as a color
import matplotlib.pyplot as plt
import matplotlib.patches as patches
fig = plt.figure()
ax = fig.add_subplot(111)
ax.add_patch(patches.Rectangle((0.1, 0.1), 0.5, 0.5, color=avg_val))
plt.show()

# print(test_array)