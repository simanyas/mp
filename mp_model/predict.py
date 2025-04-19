import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from typing import Union
import numpy as np
import gc
from tensorflow import keras
from mp_model import __version__ as _version
from mp_model.config.core import config
from mp_model.pipeline import model
from mp_model.config.core import PRED_DIR, TRAINED_MODEL_DIR

model_file_name = f"{config.app_config_.pipeline_save_file}{_version}.keras"

global label_names

def make_prediction(test_image) -> dict:
    """Make a prediction using a saved model """
    reload_path = str(TRAINED_MODEL_DIR / "mp__model_output_v0.0.1.keras")
    reloaded_model = keras.models.load_model(reload_path)
    label_names = ["partial_mask","with_mask","without_mask"]
    img_tensor = get_img_array()
    predicted = reloaded_model.predict(img_tensor)
    predicted_id = np.argmax(predicted, axis=-1)
    predicted_label = [label_names[idx] for idx in predicted_id]
    errors = False
    results = {"predictions": predicted_label, "version": _version, "errors": errors}
    print("Results:", results, predicted)
    del reloaded_model
    return results

# Function to preprocess the image into an array suitable for input into a model
def get_img_array():
    # Loading the image from the path and resizing it to the target size (180x180)
    pred_path = str(PRED_DIR / "IMG_0334.jpg")
    img = keras.utils.load_img(pred_path, target_size=(256, 256))

    # Converting the loaded image into a numpy array
    array = keras.utils.img_to_array(img)  # Converts image to a 3D numpy array (height, width, channels)

    # Adding an extra dimension to create a batch of one sample
    # This changes the shape from (height, width, channels) to (1, height, width, channels)
    array = np.expand_dims(array, axis=0)  # The shape is now (1, 180, 180, 3)

    # Returning the processed image array
    return array

if __name__ == "__main__":
    test_image="IMG_0334.jpg"
    make_prediction(test_image)
    gc.collect()
