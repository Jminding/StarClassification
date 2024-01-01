from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np
import h5py
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow import keras
from sklearn.model_selection import train_test_split
import os
import cv2
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from keras.applications.vgg16 import VGG16

df = pd.read_csv('star_data_2.csv')

sp = df["Spectral Class"]

def get_first_capital_letter(s):
    for char in s:
        if char.isalpha() and char.isupper():
            return char
    return None

sp = [get_first_capital_letter(i) for i in sp]
df["Spectral Class"] = sp

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# base_model = tf.keras.applications.MobileNetV2(input_shape=(500, 500, 3), include_top=False, weights='imagenet')
# base_model.trainable = True

# Fine-tune from this layer onwards
# fine_tune_at = 100

# # Freeze all the layers before the `fine_tune_at` layer
# for layer in base_model.layers[:fine_tune_at]:
#     layer.trainable =  False

# global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
# prediction_layer = tf.keras.layers.Dense(1)

# model = tf.keras.Sequential([
#     base_model,
#     global_average_layer,
#     prediction_layer
# ])

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(500, 500, 3))
base_model.trainable = True

# Freeze the pre-trained layers
# for layer in base_model.layers:
#     layer.trainable = False

# Add custom top layers for your classification task
model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dense(7, activation='softmax')
])

# # Create the model
# model = Sequential([
#     Conv2D(filters=32,kernel_size=(3,3),  input_shape = (500, 500, 3),activation='relu'),
#     MaxPooling2D(pool_size=(2,2)),
#
#     Conv2D(filters=32,kernel_size=(3,3), activation='relu'),
#     MaxPooling2D(pool_size=(2,2)),
#     Dropout(0.25),
#
#     Conv2D(filters=64,kernel_size=(3,3), activation='relu'),
#     MaxPooling2D(pool_size=(2,2)),
#     Dropout(0.25),
#
#     Flatten(),
#     Dense(128, activation='relu'),
#     Dropout(0.25),
#     Dense(101, activation='softmax')
# ])

base_learning_rate = 0.001
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=['accuracy'])

X_images = []
y_labels = []
classes = ['O', 'B', 'A', 'F', 'G', 'K', 'M']
star_dir = "/Users/jminding/Documents/Research/Star Images 2"
for root, dirs, files in os.walk(star_dir, topdown=False):
    for name in files:
        if name.endswith(".jpg"):
            # Read in the image_data
            image_data = cv2.imread(os.path.join(root, name))
            # Append image data to X_images
            X_images.append(image_data)
X_images = np.array(X_images)
print(X_images.shape)
for root, dirs, files in os.walk(star_dir, topdown=False):
    for name in files:
        if name.endswith(".jpg"):
            name2 = name.split("_")[0].replace("verticalflipped", "").replace(".jpg", "")
            # print(name2)
            y_labels.append(classes.index(df[df['Star Name'] == name2]['Spectral Class'].values[0][0]))
            # y_labels.append(df[df['Star Name'] == name2]['Spectral Class'].values[0][0])
y_labels = np.array(y_labels)
print(y_labels.shape)

X_train, X_test, y_train, y_test = train_test_split(X_images, y_labels, test_size=0.2, random_state=42)

callback = EarlyStopping(monitor='loss', patience=3)
history = model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test), callbacks=[callback])

model.save("model.h5")

test_loss, test_acc = model.evaluate(X_test, y_test)
print("Test accuracy:", test_acc)

# # Load the model
# model = load_model("model.h5")
#
# # Load the labels
# class_names = open("labels.txt", "r").readlines()
#
# # Create the array of the right shape to feed into the keras model
# # The 'length' or number of images you can put into the array is
# # determined by the first position in the shape tuple, in this case 1
# data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
#
# correct = 0
# one_off = 0
# incorrect = 0
#
# for i in list(df["Star Name"]):
#     # Replace this with the path to your image
#     sp_class = df[df['Star Name'] == i]['Spectral Class'].values[0][0]
#     image = Image.open(f"Star Images 2/{sp_class}/{i}.jpg").convert("RGB")
#
#     # resizing the image to be at least 224x224 and then cropping from the center
#     size = (224, 224)
#     image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
#
#     # turn the image into a numpy array
#     image_array = np.asarray(image)
#
#     # Normalize the image
#     normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
#
#     # Load the image into the array
#     data[0] = normalized_image_array
#
#     # Predicts the model
#     prediction = model.predict(data)
#     index = np.argmax(prediction)
#     class_name = class_names[index]
#     confidence_score = prediction[0][index]
#     # print(type(class_name[2:]))
#     if class_name[2:].strip() == sp_class:
#         correct += 1
#     elif (sp_class in 'FGKM' and class_name[2:].strip() in 'FGKM') or (sp_class in 'OBA' and class_name[2:].strip() in 'OBA'):
#         one_off += 1
#     else:
#         incorrect += 1
#     print(f"{i}: {class_name[2:]} ({confidence_score}), actual: {df[df['Star Name'] == i]['Spectral Class'].values[0][0]} ({'CORRECT' if class_name[2:].strip() == df[df['Star Name'] == i]['Spectral Class'].values[0][0] else 'INCORRECT'})")
#
# print(correct / (correct + incorrect + one_off))
# print((one_off + correct) / (correct + incorrect + one_off))