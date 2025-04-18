import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import pytest
from mp_model.train_pipeline import evaluate_model


def test_sample_input_data():
    evaluate_model()
    return