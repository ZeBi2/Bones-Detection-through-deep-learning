# ============================================================
# BONE FRACTURE DETECTION - STREAMLIT APP
# DenseNet121
# ============================================================

import os
import pickle
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf

from PIL import Image
from tensorflow.keras.applications.densenet import preprocess_input


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Bone Fracture Detection",
    page_icon="🦴",
    layout="centered"
)


# ============================================================
# BASE DIRECTORY
# ============================================================

BASE_DIR = Path(__file__).resolve().parent


# ============================================================
# FILE PATHS
# ============================================================

MODEL_PATH = BASE_DIR / "best_bones_fracture_model.keras"

MODEL_CONFIG_PATH = BASE_DIR / "model_config.pkl"

CLASS_MAPPING_PATH = BASE_DIR / "class_mapping.pkl"

IMAGE_CONFIG_PATH = BASE_DIR / "image_config.pkl"


# ============================================================
# HEADER
# ============================================================

st.title("🦴 Bone Fracture Detection")

st.write(
    "Upload a bone X-ray image to classify it as "
    "**fractured** or **not fractured**."
)


# ============================================================
# DEBUG / FILE CHECK
# ============================================================

required_files = {
    "Model": MODEL_PATH,
    "Model Config": MODEL_CONFIG_PATH,
    "Class Mapping": CLASS_MAPPING_PATH,
    "Image Config": IMAGE_CONFIG_PATH
}


missing_files = []

for name, path in required_files.items():

    if not path.exists():
        missing_files.append(
            f"{name}: {path.name}"
        )


# ============================================================
# STOP IF FILES ARE MISSING
# ============================================================

if missing_files:

    st.error(
        "Required deployment files are missing."
    )

    st.write(
        "Streamlit is looking in:"
    )

    st.code(
        str(BASE_DIR)
    )

    st.write(
        "Missing files:"
    )

    for item in missing_files:

        st.error(item)

    st.info(
        "Make sure all required files are uploaded "
        "to the same GitHub folder as app.py."
    )

    st.stop()


# ============================================================
# LOAD PKL CONFIGURATION
# ============================================================

@st.cache_data
def load_configurations():

    with open(
        MODEL_CONFIG_PATH,
        "rb"
    ) as file:

        model_config = pickle.load(file)


    with open(
        CLASS_MAPPING_PATH,
        "rb"
    ) as file:

        class_mapping = pickle.load(file)


    with open(
        IMAGE_CONFIG_PATH,
        "rb"
    ) as file:

        image_config = pickle.load(file)


    return (
        model_config,
        class_mapping,
        image_config
    )


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_app_model():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            "Model file not found: "
            + str(MODEL_PATH)
        )


    model = tf.keras.models.load_model(
        MODEL_PATH,

        custom_objects={
            "preprocess_input":
                preprocess_input
        },

        compile=False,

        safe_mode=False
    )


    return model


# ============================================================
# LOAD CONFIG
# ============================================================

try:

    (
        model_config,
        class_mapping,
        image_config
    ) = load_configurations()

except Exception as error:

    st.error(
        "Configuration loading failed."
    )

    st.exception(error)

    st.stop()


# ============================================================
# LOAD MODEL
# ============================================================

try:

    model = load_app_model()

except Exception as error:

    st.error(
        "Model inference failed."
    )

    st.write(
        "Please check that "
        "`best_bones_fracture_model.keras` "
        "is present and compatible with "
        "the TensorFlow version."
    )

    st.exception(error)

    st.stop()


# ============================================================
# MODEL INFORMATION
# ============================================================

classes = model_config.get(
    "classes",
    [
        "fractured",
        "not_fractured"
    ]
)


image_size = tuple(
    image_config.get(
        "image_size",
        (224, 224)
    )
)


# ============================================================
# DISPLAY MODEL STATUS
# ============================================================

st.success(
    "✅ DenseNet121 model loaded successfully."
)


# ============================================================
# IMAGE UPLOADER
# ============================================================

uploaded_file = st.file_uploader(
    "Upload Bone X-Ray Image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "bmp",
        "tif",
        "tiff"
    ]
)


# ============================================================
# PREDICTION
# ============================================================

if uploaded_file is not None:

    try:

        # ----------------------------------------------------
        # LOAD IMAGE
        # ----------------------------------------------------

        image = Image.open(
            uploaded_file
        ).convert("RGB")


        # ----------------------------------------------------
        # DISPLAY IMAGE
        # ----------------------------------------------------

        st.subheader(
            "Uploaded X-Ray"
        )

        st.image(
            image,
            caption="Uploaded X-Ray",
            use_container_width=True
        )


        # ----------------------------------------------------
        # RESIZE
        # ----------------------------------------------------

        resized_image = image.resize(
            image_size
        )


        # ----------------------------------------------------
        # CONVERT TO ARRAY
        # ----------------------------------------------------

        image_array = np.asarray(
            resized_image,
            dtype=np.float32
        )


        # ----------------------------------------------------
        # ADD BATCH DIMENSION
        # ----------------------------------------------------

        image_array = np.expand_dims(
            image_array,
            axis=0
        )


        # ----------------------------------------------------
        # MODEL PREDICTION
        # ----------------------------------------------------

        prediction = model.predict(
            image_array,
            verbose=0
        )


        # ----------------------------------------------------
        # CONVERT OUTPUT TO NUMPY
        # ----------------------------------------------------

        prediction = np.asarray(
            prediction
        )


        # ----------------------------------------------------
        # HANDLE 2-CLASS OUTPUT
        # ----------------------------------------------------

        if prediction.ndim == 2:

            probabilities = prediction[0]

        else:

            probabilities = prediction


        # ----------------------------------------------------
        # GET PREDICTED CLASS
        # ----------------------------------------------------

        predicted_index = int(
            np.argmax(
                probabilities
            )
        )


        predicted_class = classes[
            predicted_index
        ]


        confidence = float(
            probabilities[
                predicted_index
            ]
        )


        # ----------------------------------------------------
        # RESULTS
        # ----------------------------------------------------

        st.subheader(
            "Prediction Result"
        )


        if predicted_class == "fractured":

            st.error(
                "🦴 Fractured"
            )

        else:

            st.success(
                "✅ Not Fractured"
            )


        st.metric(
            "Confidence",
            f"{confidence * 100:.2f}%"
        )


        # ----------------------------------------------------
        # PROBABILITY BREAKDOWN
        # ----------------------------------------------------

        st.subheader(
            "Class Probabilities"
        )


        for index, class_name in enumerate(
            classes
        ):

            probability = float(
                probabilities[index]
            )


            st.write(
                f"{class_name}: "
                f"{probability * 100:.2f}%"
            )


            st.progress(
                min(
                    max(
                        probability,
                        0.0
                    ),
                    1.0
                )
            )


    except Exception as error:

        st.error(
            "Prediction failed."
        )

        st.exception(error)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Research and educational use only. "
    "This application is not a medical diagnostic device."
)
