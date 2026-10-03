from pathlib import Path
import pickle
import tensorflow as tf

from tensorflow.keras.applications.densenet import (
    preprocess_input
)


# ============================================================
# BASE DIRECTORY
# ============================================================

BASE_DIR = Path(__file__).resolve().parent


# ============================================================
# FILE PATHS
# ============================================================

MODEL_PATH = (
    BASE_DIR /
    "best_bones_fracture_model.keras"
)

MODEL_CONFIG_PATH = (
    BASE_DIR /
    "model_config.pkl"
)

CLASS_MAPPING_PATH = (
    BASE_DIR /
    "class_mapping.pkl"
)

IMAGE_CONFIG_PATH = (
    BASE_DIR /
    "image_config.pkl"
)


# ============================================================
# CHECK FILES
# ============================================================

required_files = [
    MODEL_PATH,
    MODEL_CONFIG_PATH,
    CLASS_MAPPING_PATH,
    IMAGE_CONFIG_PATH
]


for file_path in required_files:

    if not file_path.exists():

        raise FileNotFoundError(
            f"Required file not found: {file_path}"
        )


# ============================================================
# LOAD CONFIG
# ============================================================

with open(
    MODEL_CONFIG_PATH,
    "rb"
) as f:

    model_config = pickle.load(f)


with open(
    CLASS_MAPPING_PATH,
    "rb"
) as f:

    class_mapping = pickle.load(f)


with open(
    IMAGE_CONFIG_PATH,
    "rb"
) as f:

    image_config = pickle.load(f)


# ============================================================
# LOAD MODEL
# ============================================================

model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "preprocess_input": preprocess_input
    },
    compile=False,
    safe_mode=False
)
