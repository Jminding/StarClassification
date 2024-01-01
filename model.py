import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from PIL import Image
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import r2_score
import pandas as pd
# import h5py

df = pd.read_csv("star_data.csv")

# Define the model
def create_model():
    # Image input
    # image_input = layers.Input(shape=(500, 500, 3), name='image_input')
    image_input = layers.Input(shape=(1,), name='image_input')

    # Luminosity input
    luminosity_input = layers.Input(shape=(1,), name='luminosity_input')

    # Flatten the image and concatenate with the luminosity input
    flat_image = layers.Flatten()(image_input)
    concatenated_input = layers.Concatenate()([flat_image, luminosity_input])

    # Dense layers for correlation
    dense1 = layers.Dense(64, activation='relu')(concatenated_input)
    dense3 = layers.Dense(64, activation='relu')(dense1)
    dense2 = layers.Dense(32, activation='relu')(dense3)

    # Output layer for distance prediction
    distance_output = layers.Dense(1, name='distance_output')(dense2)

    # Define the model with two inputs and one output
    model = models.Model(inputs=[image_input, luminosity_input], outputs=distance_output)

    return model

num_samples = 1000
height, width, channels = 500, 500, 3

# Define the path to your image directory
image_dir = 'Star Images/'

# Get a list of all image file names in the directory
image_files = [(f + "_removedbg.jpg") for f in list(df["Star Name"])]

# Define the size of your images
target_size = (500, 500)  # Adjust as needed

# Create empty arrays to store the images and luminosity values
X_images = []
X_luminosity = list(df["L/L_o"])

# Load images and extract luminosity values
for image_file in image_files:
    # Load image using PIL
    img_path = os.path.join(image_dir, image_file)
    img = Image.open(img_path)
    img = img.resize(target_size)  # Resize image if needed

    # Convert image to NumPy array
    img_array = image.img_to_array(img)

    avg = np.average(img_array)

    # Append data to arrays
    X_images.append(avg)

# Convert lists to NumPy arrays
X_images = np.array(X_images)
X_luminosity = np.array(X_luminosity)

# # Normalize image pixel values to the range [0, 1]
# X_images /= 255.0

# X_images = np.random.rand(num_samples, height, width, channels)
# X_luminosity = np.random.rand(num_samples, 1)
y_distance = np.array(list(df["Distance (pc)"]))

# Create an instance of the model
model = create_model()

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Display the model summary
model.summary()

# Train the model
model.fit([X_images, X_luminosity], y_distance, epochs=10, batch_size=32)

# Find the model accuracy
y_pred = np.abs(model.predict([X_images, X_luminosity]))
print(list(np.abs(y_distance - y_pred) / y_distance))
# print()
r2 = r2_score(y_distance, y_pred, force_finite=True)
print("R2 score: {}".format(r2))

# Graph the results
import matplotlib.pyplot as plt
plt.scatter(1 / y_distance, 1 / y_pred)
plt.xlabel("Actual Distance (pc)")
plt.ylabel("Predicted Distance (pc)")
plt.title("Distance Prediction")
# Only show less than 2000
plt.xlim(0, 0.1)
plt.ylim(0, 1)
plt.show()

# Save the model as a file
# model.save('star_distance_model.h5')
