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

model = load_model('keras_model.h5')
print(model.summary())
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
