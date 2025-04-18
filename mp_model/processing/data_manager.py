import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import typing as t
from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

import glob
from PIL import Image
from collections import defaultdict
from pathlib import Path

from mp_model import __version__ as _version
from mp_model.config.core import DATASET_DIR, TRAINED_MODEL_DIR, PRED_DIR, config


global paths, shapes, modes, format

def size_image(source_path: str) -> None:
    shapes = defaultdict(int) # Initialize with int
    modes = defaultdict(int) # Initialize with int
    format = defaultdict(int) # Initialize with int
    #print(source_path)
    with Image.open(source_path) as img:
        shapes[(img.size[0], img.size[1])] += 1
        modes[img.mode] += 1
        format[img.format] += 1


def preprocess_training_data():
    paths = []
    paths.append(DATASET_DIR + "MP2_FaceMask_Dataset" + "train" + "/with_mask/*")
    paths.append(DATASET_DIR + "MP2_FaceMask_Dataset" + "train" + "/without_mask/*")
    paths.append(DATASET_DIR + "MP2_FaceMask_Dataset" + "train" + "/partial_mask/*")
    paths.append(DATASET_DIR + "MP2_FaceMask_Dataset" + "test" + "/with_mask/*")
    paths.append(DATASET_DIR + "MP2_FaceMask_Dataset" + "test" + "/without_mask/*")
    paths.append(DATASET_DIR + "MP2_FaceMask_Dataset" + "test" + "/partial_mask/*")
    for path_list in paths:
        for path in glob.glob(path_list):
            size_image(path)
    mode_shapes = defaultdict(int)
    for k,v in shapes.items():
        if v>25:
            mode_shapes[k] = v
    print(format)
    print(shapes)
    print(modes)
    

##  Pre-Pipeline Preparation
def save_model(model) -> None:
    """Persist the pipeline.
    Saves the versioned model, and overwrites any previous saved models. 
    This ensures that when the package is published, there is only one trained model that 
    can be called, and we know exactly how it was built.
    """
    # Prepare versioned save file name
    save_file_name = f"{config.app_config_.pipeline_save_file}{_version}.keras"
    save_path = TRAINED_MODEL_DIR / save_file_name
    #joblib.dump(pipeline_to_persist, save_path)
    model.save(save_path)
    print("Model saved successfully.")
