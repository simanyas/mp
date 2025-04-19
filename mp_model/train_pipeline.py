import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from keras.layers import Conv2D, Input, Activation, MaxPooling2D, Dense
from keras.models import Model, load_model

import numpy as np
import PIL
import tensorflow as tf
from tensorflow import keras
import glob, os
from tensorflow.keras import layers

from mp_model.config.core import config
from mp_model.pipeline import model
from mp_model.processing.data_manager import save_model, preprocess_training_data
from mp_model.config.core import DATASET_DIR, TRAINED_MODEL_DIR

global train_generator, val_generator, train_path, test_path

def evaluate_model():
    test_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    val_generator = test_image_generator.flow_from_directory(test_path,
                                            class_mode='categorical',
                                            shuffle=False,
                                            batch_size=32
                                           )
    reload_path = str(TRAINED_MODEL_DIR / "mp__model_output_v0.0.1.keras")
    reloaded_model = tf.keras.models.load_model(reload_path)
    score = reloaded_model.evaluate(val_generator)
    print(f'Test loss: {score[0]} / Test f1_score: {score[1]}')
    return score

def run_training() -> None:
    """
    Train the model.
    """
    # Compile the model

    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='categorical_crossentropy', metrics=['f1_score'])
    history = model.fit(train_generator, epochs=1, validation_data=val_generator)
    # read training data
    #data = load_dataset(file_name = config.app_config_.training_data_file)
    
    score = evaluate_model()
    # persist trained model
    save_model(model)


if __name__ == "__main__":
    preprocess_training_data()
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
    train_path = str(DATASET_DIR / "MP2_FaceMask_Dataset" / "train")
    print(train_path)
    train_generator = img_gen.flow_from_directory(train_path,
                                              class_mode='categorical',
                                              subset='training',
                                              shuffle=True,
                                              batch_size=32
                                             )
    # Validation data
    test_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    test_path = str(DATASET_DIR / "MP2_FaceMask_Dataset" / "test")
    print(test_path)
    val_generator = test_image_generator.flow_from_directory(test_path,
                                            class_mode='categorical',
                                            shuffle=False,
                                            batch_size=32
                                           )
    for image_batch, label_batch in train_generator:
        print("Image batch shape: ", image_batch.shape)
        print("Label batch shape: ", label_batch.shape)
        break
    #batch_stats_callback = CollectBatchStats()
    run_training()
