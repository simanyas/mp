import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import json
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException, Body
from fastapi.encoders import jsonable_encoder
from mp_model.config.core import TRAINED_MODEL_DIR, PRED_DIR
from mp_model import __version__ as model_version
from tensorflow import keras
from app import __version__, schemas
from app.config import settings

api_router = APIRouter()


@api_router.get("/health", response_model=schemas.Health, status_code=200)
def health() -> dict:
    """
    Root Get
    """
    health = schemas.Health(
        name=settings.PROJECT_NAME, api_version=__version__, model_version=model_version
    )

    return health.dict()



example_input = {
    "inputs": [
        {
            "predict_image": "IMG_0334.jpg"
        }
    ]
}
def make_prediction(test_image) -> dict:
    """Make a prediction using a pre-trained, saved model """
    reloaded_model = keras.models.load_model(str(TRAINED_MODEL_DIR / "mp__model_output_v0.0.1.keras"))
    label_names = ["partial_mask","with_mask","without_mask"]
    img_tensor = get_img_array()
    predicted = reloaded_model.predict(img_tensor)
    predicted_id = np.argmax(predicted, axis=-1)
    predicted_label = [label_names[idx] for idx in predicted_id]
    errors = False
    results = {"predictions": predicted_label[0], "version": model_version, "errors": errors}
    print("Results:", results, predicted)
    return results

# Function to preprocess the image into an array suitable for input into a model
def get_img_array():
    # Loading the image from the path and resizing it to the target size (180x180)
    img = keras.utils.load_img(str(PRED_DIR / "IMG_0334.jpg"), target_size=(256, 256))

    # Converting the loaded image into a numpy array
    array = keras.utils.img_to_array(img)  # Converts image to a 3D numpy array (height, width, channels)

    # Adding an extra dimension to create a batch of one sample
    # This changes the shape from (height, width, channels) to (1, height, width, channels)
    array = np.expand_dims(array, axis=0)  # The shape is now (1, 180, 180, 3)

    # Returning the processed image array
    return array

@api_router.post("/predict", response_model=schemas.PredictionResults, status_code=200)
async def predict(input_data: schemas.MultipleDataInputs = Body(..., example=example_input)) -> Any:
    """
    Mask prediction with the mp_model
    """
    #input_image_name = jsonable_encoder(input_data.inputs)
    #print("input_image_name = ", input_image_name)
    results = make_prediction("IMG_0334.jpg")

    if results["errors"] is not False:
        raise HTTPException(status_code=400, detail=json.loads(results["errors"]))

    return results
