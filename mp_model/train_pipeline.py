import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from keras.models import Sequential
from keras.layers import Conv2D, Input, ZeroPadding2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense,Dropout
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint

from sklearn.metrics import f1_score
from sklearn.utils import shuffle

import numpy as np
import pandas as pd
import PIL
from keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50
import tensorflow as tf
from tensorflow import keras
import glob, os
from tensorflow.keras import layers

from mp_model.config.core import config
from mp_model.pipeline import model
from mp_model.processing.data_manager import save_model
from mp_model.processing.features import CollectBatchStats
from mp_model.config.core import DATASET_DIR, TRAINED_MODEL_DIR

global train_generator, val_generator, batch_stats_callback

def evaluate_model():
    test_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    val_generator = test_image_generator.flow_from_directory(DATASET_DIR / "MP2_FaceMask_Dataset" / "test",
                                            class_mode='categorical',
                                            shuffle=False,
                                            batch_size=32
                                           )
    #reloaded_model = tf.keras.models.load_model(TRAINED_MODEL_DIR / "mp__model_output_v0.0.1.keras")
    score = model.evaluate(val_generator)
    print(f'Test loss: {score[0]} / Test f1_score: {score[1]}')
    return score

def run_training() -> None:
    """
    Train the model.
    """
    # Compile the model
    '''
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='categorical_crossentropy', metrics=['f1_score'])
    history = model.fit(train_generator, epochs=1, validation_data=val_generator, callbacks = [batch_stats_callback])
    # read training data
    #data = load_dataset(file_name = config.app_config_.training_data_file)
    
    score = evaluate_model()
    # persist trained model
    save_model(model)
    '''

if __name__ == "__main__":
    '''
    img_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
                                                          rotation_range=20,
                                                          horizontal_flip=True,
                                                          validation_split=0.2,
                                                          height_shift_range=0.1,
                                                          width_shift_range=0.1,
                                                          brightness_range=(0.5,1.5),
                                                          shear_range = 0.20,
                                                          zoom_range = [1, 1.5],
                                                          fill_mode='nearest'
                                                         )
    train_generator = img_gen.flow_from_directory(DATASET_DIR / "MP2_FaceMask_Dataset" / "train",
                                              class_mode='categorical',
                                              subset='training',
                                              shuffle=True,
                                              batch_size=32
                                             )
    # Validation data
    test_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    val_generator = test_image_generator.flow_from_directory(DATASET_DIR / "MP2_FaceMask_Dataset" / "test",
                                            class_mode='categorical',
                                            shuffle=False,
                                            batch_size=32
                                           )
    for image_batch, label_batch in train_generator:
        print("Image batch shape: ", image_batch.shape)
        print("Label batch shape: ", label_batch.shape)
        break
    batch_stats_callback = CollectBatchStats()
    '''
    run_training()
