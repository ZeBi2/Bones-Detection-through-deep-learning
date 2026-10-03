import json
import pickle
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Bone Fracture Detection",
    page_icon="🦴",
    layout="centered"
)


# ============================================================
# PROJECT DIRECTORY
# ============================================================

BASE_DIR = Path(__file__).resolve().parent


# ============================================================
# DEPLOYMENT FILES
# ============================================================

MODEL_PATH = BASE_DIR / "best_bones_fracture_model.keras"

MODEL_CONFIG_PATH = BASE_DIR / "model_config.pkl"

CLASS_MAPPING_PATH = BASE_DIR / "class_mapping.pkl"

IMAGE_CONFIG_PATH = BASE_DIR / "image_config.pkl"

METADATA_JSON_PATH = BASE_DIR / "best_model_metadata.json"

CLASS_MAPPING_JSON_PATH = BASE_DIR / "class_mapping.json"


# ============================================================
# APPLICATION TITLE
# ============================================================

st.title("🦴 Bone Fracture Detection")

st.markdown(
    """
    Upload a bone X-ray image and the trained DenseNet121
    model will classify it as **Fractured** or
    **Not Fractured**.
    """
)


# ============================================================
# REQUIRED FILES
# ============================================================

REQUIRED_FILES = {
    "Model": MODEL_PATH,
    "Model configuration": MODEL_CONFIG_PATH,
    "Class mapping": CLASS_MAPPING_PATH,
    "Image configuration": IMAGE_CONFIG_PATH,
}


# ============================================================
# FILE VALIDATION
# ============================================================

missing_files = []

for file_name, file_path in REQUIRED_FILES.items():

    if not file_path.is_file():

        missing_files.append(
            f"{file_name}: {file_path.name}"
        )


if missing_files:

    st.error(
        "❌ Required deployment files are missing."
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

    for missing in missing_files:

        st.error(
            missing
        )

    st.warning(
        "Make sure all required files are uploaded "
        "to the same GitHub directory as app.py."
    )

    st.stop()


# ============================================================
# LOAD PKL CONFIGURATIONS
# ============================================================

try:

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


except Exception as error:

    st.error(
        "❌ Failed to load deployment configuration."
    )

    st.exception(error)

    st.stop()


# ============================================================
# LOAD OPTIONAL JSON METADATA
# ============================================================

metadata_json = None

if METADATA_JSON_PATH.is_file():

    try:

        with open(
            METADATA_JSON_PATH,
            "r",
            encoding="utf-8"
        ) as file:

            metadata_json = json.load(file)

    except Exception:

        metadata_json = None


# ============================================================
# CLASSES
# ============================================================

classes = model_config.get(
    "classes",
    [
        "fractured",
        "not_fractured"
    ]
)


# ============================================================
# CLASS MAPPING
# ============================================================

if not class_mapping:

    class_mapping = {
        "fractured": 0,
        "not_fractured": 1
    }


# ============================================================
# IMAGE SIZE
# ============================================================

image_size = tuple(
    model_config.get(
        "image_size",
        image_config.get(
            "image_size",
            (224, 224)
        )
    )
)


# ============================================================
# MODEL INFORMATION
# ============================================================

MODEL_NAME = model_config.get(
    "model_name",
    "DenseNet121"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model(model_path):

    return tf.keras.models.load_model(
        str(model_path),
        compile=False,
        safe_mode=False
    )


# ============================================================
# LOAD MODEL
# ============================================================

try:

    with st.spinner(
        "Loading DenseNet121 model..."
    ):

        model = load_model(
            MODEL_PATH
        )

except Exception as error:

    st.error(
        "❌ Model loading failed."
    )

    st.write(
        "The model file exists, but TensorFlow/Keras "
        "could not load it."
    )

    st.exception(error)

    st.stop()


# ============================================================
# VERIFY MODEL
# ============================================================

try:

    model_input_shape = model.input_shape

    model_output_shape = model.output_shape

except Exception as error:

    st.error(
        "❌ Could not read model input/output shape."
    )

    st.exception(error)

    st.stop()


# ============================================================
# VERIFY OUTPUT CLASSES
# ============================================================

try:

    number_of_outputs = int(
        model_output_shape[-1]
    )

except Exception:

    number_of_outputs = 2


if number_of_outputs != len(classes):

    st.error(
        "❌ Model/configuration mismatch."
    )

    st.write(
        f"Model outputs: {number_of_outputs}"
    )

    st.write(
        f"Configured classes: {len(classes)}"
    )

    st.stop()


# ============================================================
# MODEL READY
# ============================================================

st.success(
    f"✅ {MODEL_NAME} model loaded successfully."
)


# ============================================================
# MODEL DETAILS
# ============================================================

with st.expander(
    "Model Information"
):

    st.write(
        "**Architecture:**",
        MODEL_NAME
    )

    st.write(
        "**Input Shape:**",
        model_input_shape
    )

    st.write(
        "**Output Shape:**",
        model_output_shape
    )

    st.write(
        "**Image Size:**",
        image_size
    )

    st.write(
        "**Classes:**",
        classes
    )

    st.write(
        "**Class Mapping:**",
        class_mapping
    )

    if metadata_json:

        if "best_f1_score" in metadata_json:

            st.write(
                "**Best F1-Score:**",
                f"{metadata_json['best_f1_score']:.4f}"
            )


# ============================================================
# IMAGE UPLOADER
# ============================================================

st.subheader(
    "Upload X-Ray Image"
)


uploaded_file = st.file_uploader(
    "Choose an X-ray image",
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
        # OPEN IMAGE
        # ----------------------------------------------------

        image = Image.open(
            uploaded_file
        ).convert("RGB")


        # ----------------------------------------------------
        # DISPLAY ORIGINAL IMAGE
        # ----------------------------------------------------

        st.subheader(
            "Uploaded X-Ray"
        )

        st.image(
            image,
            caption="Uploaded Bone X-Ray",
            use_container_width=True
        )


        # ----------------------------------------------------
        # RESIZE IMAGE
        # ----------------------------------------------------

        resized_image = image.resize(
            image_size
        )


        # ----------------------------------------------------
        # CONVERT TO NUMPY
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
        # IMPORTANT
        # ----------------------------------------------------
        # The saved DenseNet121 model already contains
        # DenseNet preprocess_input.
        #
        # Therefore:
        #
        # DO NOT call preprocess_input() here.
        #
        # Raw RGB image is passed to the model.
        # ----------------------------------------------------


        # ----------------------------------------------------
        # MODEL PREDICTION
        # ----------------------------------------------------

        with st.spinner(
            "Analyzing X-ray..."
        ):

            prediction = model.predict(
                image_array,
                verbose=0
            )


        # ----------------------------------------------------
        # CONVERT PREDICTION
        # ----------------------------------------------------

        prediction = np.asarray(
            prediction
        )


        # ----------------------------------------------------
        # GET PROBABILITIES
        # ----------------------------------------------------

        if prediction.ndim == 2:

            probabilities = prediction[0]

        else:

            probabilities = prediction.flatten()


        # ----------------------------------------------------
        # VALIDATE OUTPUT
        # ----------------------------------------------------

        if len(probabilities) != len(classes):

            raise ValueError(
                "Model prediction output does not "
                "match configured classes."
            )


        # ----------------------------------------------------
        # PREDICTED INDEX
        # ----------------------------------------------------

        predicted_index = int(
            np.argmax(
                probabilities
            )
        )


        # ----------------------------------------------------
        # PREDICTED CLASS
        # ----------------------------------------------------

        predicted_class = classes[
            predicted_index
        ]


        # ----------------------------------------------------
        # CONFIDENCE
        # ----------------------------------------------------

        confidence = float(
            probabilities[
                predicted_index
            ]
        )


        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        st.divider()

        st.subheader(
            "Prediction Result"
        )


        if predicted_class == "fractured":

            st.error(
                "🦴 FRACTURED"
            )

        else:

            st.success(
                "✅ NOT FRACTURED"
            )


        st.metric(
            "Confidence",
            f"{confidence * 100:.2f}%"
        )


        # ----------------------------------------------------
        # CLASS PROBABILITIES
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


            display_name = class_name.replace(
                "_",
                " "
            ).title()


            st.write(
                f"**{display_name}: "
                f"{probability * 100:.2f}%**"
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
            "❌ Prediction failed."
        )

        st.exception(error)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Research and educational use only. "
    "This application is not a medical diagnostic device. "
    "Please consult a qualified healthcare professional "
    "for medical diagnosis."
)
