import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from keras.models import Sequential
from keras.layers import Conv2D, Input, Activation, MaxPooling2D, Dense
from keras.models import Model, load_model
import tensorflow as tf
from tensorflow import keras
import glob, os
from tensorflow.keras import layers

from mp_model.config.core import config

def create_model():
    # Input layer: Defines the input shape for the model. It expects a 28x28 pixel image with 1 channel (grayscale).
    inputs = keras.Input(shape=(256, 256, 3))

    # First convolutional layer: Applies 32 filters of size 3x3 with ReLU activation function to extract features from the input.
    x = layers.Conv2D(filters=128, kernel_size=3, activation="relu")(inputs)

    # First max-pooling layer: Reduces the spatial dimensions of the feature map by taking the maximum value over a 2x2 window.
    x = layers.MaxPooling2D()(x)

    # Second convolutional layer: Applies 64 filters of size 3x3 with ReLU activation to further extract features.
    x = layers.Conv2D(filters=256, kernel_size=3, activation="relu")(x)

    # Second max-pooling layer: Reduces the spatial dimensions further with a 2x2 window.
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(filters=512, kernel_size=3, activation="relu")(x)

    # Second max-pooling layer: Reduces the spatial dimensions further with a 2x2 window.
    x = tf.keras.layers.GlobalMaxPooling2D()(x)

    # Output layer: A dense layer with 10 units (for 10 classes) and softmax activation function to output probability distribution over the 10 classes.
    outputs = layers.Dense(3, activation="softmax")(x)

    # Define the model: Specifies the input and output layers for the model.
    model = keras.Model(inputs=inputs, outputs=outputs)
    print(model.summary())
    return model

model = create_model()
