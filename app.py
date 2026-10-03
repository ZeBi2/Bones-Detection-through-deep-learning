import os
import pickle
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf

from PIL import Image
from tensorflow.keras.applications.densenet import preprocess_input


# ============================================================
# PAGE CONFIGURATION
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
    "Model Configuration": MODEL_CONFIG_PATH,
    "Class Mapping": CLASS_MAPPING_PATH,
    "Image Configuration": IMAGE_CONFIG_PATH
}


missing_files = []

for file_name, file_path in required_files.items():

    if not file_path.exists():

        missing_files.append(
            f"{file_name}: {file_path.name}"
        )


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

        st.error(
            item
        )

    st.info(
        "Make sure these files are uploaded to the "
        "same GitHub folder as app.py."
    )

    st.stop()


# ============================================================
# LOAD MODEL CONFIGURATION
# ============================================================

try:

    with open(
        MODEL_CONFIG_PATH,
        "rb"
    ) as file:

        model_config = pickle.load(file)

except Exception as error:

    st.error(
        "Could not load model_config.pkl"
    )

    st.exception(error)

    st.stop()


# ============================================================
# LOAD CLASS MAPPING
# ============================================================

try:

    with open(
        CLASS_MAPPING_PATH,
        "rb"
    ) as file:

        class_mapping = pickle.load(file)

except Exception as error:

    st.error(
        "Could not load class_mapping.pkl"
    )

    st.exception(error)

    st.stop()


# ============================================================
# LOAD IMAGE CONFIGURATION
# ============================================================

try:

    with open(
        IMAGE_CONFIG_PATH,
        "rb"
    ) as file:

        image_config = pickle.load(file)

except Exception as error:

    st.error(
        "Could not load image_config.pkl"
    )

    st.exception(error)

    st.stop()


# ============================================================
# IMAGE SIZE
# ============================================================

image_size = tuple(
    image_config.get(
        "image_size",
        (224, 224)
    )
)


# ============================================================
# CLASS INFORMATION
# ============================================================

classes = model_config.get(
    "classes",
    [
        "fractured",
        "not_fractured"
    ]
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    return tf.keras.models.load_model(

        MODEL_PATH,

        custom_objects={
            "preprocess_input":
                preprocess_input
        },

        compile=False,

        safe_mode=False
    )


try:

    with st.spinner(
        "Loading AI model..."
    ):

        model = load_model()

except Exception as error:

    st.error(
        "Model inference failed."
    )

    st.write(
        "The model file was found, but TensorFlow "
        "could not load it."
    )

    st.exception(error)

    st.stop()


# ============================================================
# MODEL READY
# ============================================================

st.success(
    "✅ DenseNet121 model loaded successfully."
)


# ============================================================
# UPLOAD IMAGE
# ============================================================

uploaded_file = st.file_uploader(

    "Upload a bone X-ray image",

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
        # DISPLAY IMAGE
        # ----------------------------------------------------

        st.subheader(
            "Uploaded X-ray"
        )

        st.image(
            image,
            caption="Uploaded X-ray",
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
        # PREDICTION
        # ----------------------------------------------------

        prediction = model.predict(
            image_array,
            verbose=0
        )


        # ----------------------------------------------------
        # HANDLE TWO-CLASS OUTPUT
        # ----------------------------------------------------

        prediction = np.asarray(
            prediction
        )


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


        # ----------------------------------------------------
        # GET CLASS NAME
        # ----------------------------------------------------

        index_to_class = {
            int(index): class_name
            for class_name, index
            in class_mapping.items()
        }


        predicted_class = index_to_class.get(
            predicted_index,
            classes[predicted_index]
        )


        confidence = float(
            probabilities[predicted_index]
        ) * 100


        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        st.subheader(
            "Prediction Result"
        )


        if predicted_class == "fractured":

            st.error(
                f"🦴 Fractured\n\n"
                f"Confidence: {confidence:.2f}%"
            )

        else:

            st.success(
                f"✅ Not Fractured\n\n"
                f"Confidence: {confidence:.2f}%"
            )


        # ----------------------------------------------------
        # CLASS PROBABILITIES
        # ----------------------------------------------------

        st.subheader(
            "Class Probabilities"
        )


        for index, class_name in index_to_class.items():

            if index < len(probabilities):

                probability = (
                    float(
                        probabilities[index]
                    ) * 100
                )

                st.write(
                    f"{class_name}: "
                    f"{probability:.2f}%"
                )

                st.progress(
                    min(
                        max(
                            probability / 100,
                            0.0
                        ),
                        1.0
                    )
                )


        # ----------------------------------------------------
        # MODEL INFORMATION
        # ----------------------------------------------------

        with st.expander(
            "Model Information"
        ):

            st.write(
                "Model:",
                model_config.get(
                    "model_name",
                    "DenseNet121"
                )
            )

            st.write(
                "Architecture:",
                model_config.get(
                    "model_architecture",
                    "DenseNet121"
                )
            )

            st.write(
                "Image Size:",
                image_size
            )

            st.write(
                "Classes:",
                classes
            )


    except Exception as error:

        st.error(
            "Prediction failed."
        )

        st.exception(error)


# ============================================================
# DISCLAIMER
# ============================================================

st.markdown(
    "---"
)

st.caption(
    "This application is intended for research and "
    "educational purposes only. It is not a medical "
    "diagnostic device and should not replace assessment "
    "by a qualified healthcare professional."
)
