import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
import h5py
import numpy as np
import os
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from keras import backend as K
from tensorflow.keras.utils import plot_model

df = pd.read_csv('star_data_2.csv')

sp = df["Spectral Class"]

def get_first_capital_letter(s):
    for char in s:
        if char.isalpha() and char.isupper():
            return char
    return None

sp = [get_first_capital_letter(i) for i in sp]
df["Spectral Class"] = sp

# X_images = []
# y_labels = []
# classes = ['O', 'B', 'A', 'F', 'G', 'K', 'M']
# star_dir = "/Users/jminding/Documents/Research/Star Images 2"
# for root, dirs, files in os.walk(star_dir, topdown=False):
#     for name in files:
#         if name.endswith(".jpg"):
#             # Read in the image_data
#             image_data = cv2.imread(os.path.join(root, name))
#             # Reshape image to 224x224
#             image_data = cv2.resize(image_data, (224, 224))
#             # Append image data to X_images
#             X_images.append(image_data)
# X_images = np.array(X_images)
# print(X_images.shape)
# for root, dirs, files in os.walk(star_dir, topdown=False):
#     for name in files:
#         if name.endswith(".jpg"):
#             name2 = name.split("_")[0].replace("verticalflipped", "").replace(".jpg", "")
#             # print(name2)
#             y_labels.append(classes.index(df[df['Star Name'] == name2]['Spectral Class'].values[0][0]))
#             # y_labels.append(df[df['Star Name'] == name2]['Spectral Class'].values[0][0])
# y_labels = np.array(y_labels)
# print(y_labels.shape)
#
# X_train, X_test, y_train, y_test = train_test_split(X_images, y_labels, test_size=0.2, random_state=42)
#
# model = models.load_model("keras_model.h5")
# model.compile(optimizer="adam",
#                 loss="categorical_crossentropy",
#                 metrics=['accuracy'])
# test_loss, test_acc = model.evaluate(X_test, y_test)
# print("Test accuracy:", test_acc)


# Load the model
model = load_model("keras_model.h5")

model.summary()

# visualize the model
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

exit()

# Load the labels
class_names = open("labels.txt", "r").readlines()

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

correct = 0
one_off = 0
incorrect = 0

# K.set_value(model.optimizer.learning_rate, 0.001)

X_images = []
y_classes = []
classes = list("OBAFGKM")
for i in list(df["Star Name"]):
    for j in [i, f"{i}_90_180", f"{i}_90", f"{i}_180_180", f"{i}_180", f"{i}_270", f"{i}_270_180", f"{i}_horizontalflipped_90_180", f"{i}_horizontalflipped_90", f"{i}_horizontalflipped_180_180", f"{i}_horizontalflipped_180", f"{i}_horizontalflipped_270_180", f"{i}_horizontalflipped_270", f"{i}_horizontalflipped"]:
        # Replace this with the path to your image
        sp_class = df[df['Star Name'] == i]['Spectral Class'].values[0][0]
        image = Image.open(f"Star Images 2/{sp_class}/{j}.jpg").convert("RGB")

        # resizing the image to be at least 224x224 and then cropping from the center
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

        # turn the image into a numpy array
        image_array = np.array(image)

        # Normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

        # Turn the image back into a list
        normalized_image_array = normalized_image_array.tolist()

        # Load the image into the array
        data[0] = normalized_image_array
        X_images.append(normalized_image_array)
        y_classes.append((classes.index(sp_class), sp_class))
        print(j)

print("Compiling (this takes a long time)")
model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001), loss="categorical_crossentropy", metrics=['accuracy'])
model.fit(X_images, y_classes, epochs=50, batch_size=16)

for i in list(df["Star Name"]):
    for j in [i, f"{i}_90_180", f"{i}_90", f"{i}_180_180", f"{i}_180", f"{i}_270", f"{i}_270_180", f"{i}_horizontalflipped_90_180", f"{i}_horizontalflipped_90", f"{i}_horizontalflipped_180_180", f"{i}_horizontalflipped_180", f"{i}_horizontalflipped_270_180", f"{i}_horizontalflipped_270", f"{i}_horizontalflipped"]:
        # Predicts the model
        sp_class = df[df['Star Name'] == i]['Spectral Class'].values[0][0]
        image = Image.open(f"Star Images 2/{sp_class}/{j}.jpg").convert("RGB")

        # resizing the image to be at least 224x224 and then cropping from the center
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

        # turn the image into a numpy array
        image_array = np.asarray(image)

        # Normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

        # Load the image into the array
        data[0] = normalized_image_array
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]
        predictions = {class_names[i].strip(): prediction[0][i] for i in range(len(class_names))}
        print(j)
        print(predictions)
        exit()
        # print(type(class_name[2:]))
        if class_name[2:].strip() == sp_class:
            correct += 1
        elif (sp_class in 'FGKM' and class_name[2:].strip() in 'FGKM') or (sp_class in 'OBA' and class_name[2:].strip() in 'OBA'):
            one_off += 1
        else:
            incorrect += 1
        print(f"{i}: {class_name[2:]} ({confidence_score}), actual: {df[df['Star Name'] == i]['Spectral Class'].values[0][0]} ({'CORRECT' if class_name[2:].strip() == df[df['Star Name'] == i]['Spectral Class'].values[0][0] else 'INCORRECT'})")

print(correct / (correct + incorrect + one_off))
print((one_off + correct) / (correct + incorrect + one_off))
