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
# FIND FILE
# ============================================================

def find_file(filename):

    possible_paths = [

        BASE_DIR / filename,

        BASE_DIR / "models" / filename,

        BASE_DIR / "model" / filename,

        BASE_DIR / "weights" / filename,

    ]

    for path in possible_paths:

        if path.exists():
            return path

    return None


# ============================================================
# FIND DEPLOYMENT FILES
# ============================================================

MODEL_PATH = find_file(
    "best_bones_fracture_model.keras"
)

MODEL_CONFIG_PATH = find_file(
    "model_config.pkl"
)

CLASS_MAPPING_PATH = find_file(
    "class_mapping.pkl"
)

IMAGE_CONFIG_PATH = find_file(
    "image_config.pkl"
)


# ============================================================
# TITLE
# ============================================================

st.title(
    "🦴 Bone Fracture Detection"
)

st.write(
    "Upload a bone X-ray image and the trained "
    "DenseNet121 model will classify it as "
    "**Fractured** or **Not Fractured**."
)


# ============================================================
# REQUIRED FILE CHECK
# ============================================================

missing_files = []


if MODEL_PATH is None:

    missing_files.append(
        "best_bones_fracture_model.keras"
    )


if MODEL_CONFIG_PATH is None:

    missing_files.append(
        "model_config.pkl"
    )


if CLASS_MAPPING_PATH is None:

    missing_files.append(
        "class_mapping.pkl"
    )


if IMAGE_CONFIG_PATH is None:

    missing_files.append(
        "image_config.pkl"
    )


if missing_files:

    st.error(
        "❌ Required deployment files are missing."
    )

    st.write(
        "Streamlit application directory:"
    )

    st.code(
        str(BASE_DIR)
    )

    st.write(
        "Missing files:"
    )

    for filename in missing_files:

        st.error(
            filename
        )

    st.warning(
        "The trained .keras model must be uploaded "
        "to the GitHub repository. PKL files cannot "
        "replace the trained model."
    )

    st.info(
        "Required model filename: "
        "best_bones_fracture_model.keras"
    )

    st.stop()


# ============================================================
# LOAD CONFIGURATION
# ============================================================

try:

    with open(
        MODEL_CONFIG_PATH,
        "rb"
    ) as file:

        model_config = pickle.load(
            file
        )


    with open(
        CLASS_MAPPING_PATH,
        "rb"
    ) as file:

        class_mapping = pickle.load(
            file
        )


    with open(
        IMAGE_CONFIG_PATH,
        "rb"
    ) as file:

        image_config = pickle.load(
            file
        )


except Exception as error:

    st.error(
        "❌ Configuration loading failed."
    )

    st.exception(
        error
    )

    st.stop()


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model(model_path):

    return tf.keras.models.load_model(

        str(model_path),

        custom_objects={
            "preprocess_input":
                preprocess_input
        },

        compile=False,

        safe_mode=False
    )


# ============================================================
# MODEL LOADING
# ============================================================

try:

    model = load_model(
        MODEL_PATH
    )

except Exception as error:

    st.error(
        "❌ Model loading failed."
    )

    st.error(
        "The model file exists but could not "
        "be loaded."
    )

    st.exception(
        error
    )

    st.stop()


# ============================================================
# MODEL STATUS
# ============================================================

st.success(
    "✅ DenseNet121 model loaded successfully."
)


# ============================================================
# GET MODEL CONFIG
# ============================================================

classes = model_config.get(

    "classes",

    [
        "fractured",
        "not_fractured"
    ]
)


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
# DISPLAY MODEL INFORMATION
# ============================================================

with st.expander(
    "Model Information"
):

    st.write(
        "**Architecture:** DenseNet121"
    )

    st.write(
        f"**Input Size:** "
        f"{image_size[0]} × {image_size[1]}"
    )

    st.write(
        "**Classes:**"
    )

    st.write(
        classes
    )


# ============================================================
# IMAGE UPLOAD
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
        # READ IMAGE
        # ----------------------------------------------------

        image = Image.open(
            uploaded_file
        ).convert(
            "RGB"
        )


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
        # ARRAY
        # ----------------------------------------------------

        image_array = np.asarray(

            resized_image,

            dtype=np.float32
        )


        # ----------------------------------------------------
        # BATCH
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


        prediction = np.asarray(
            prediction
        )


        # ----------------------------------------------------
        # TWO CLASS OUTPUT
        # ----------------------------------------------------

        if prediction.ndim == 2:

            probabilities = prediction[0]

        else:

            probabilities = prediction


        # ----------------------------------------------------
        # CHECK OUTPUT
        # ----------------------------------------------------

        if len(probabilities) != len(classes):

            raise ValueError(

                "Model output does not match "
                "the configured number of classes."
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
        # PROBABILITIES
        # ----------------------------------------------------

        st.subheader(
            "Class Probabilities"
        )


        for index, class_name in enumerate(
            classes
        ):

            probability = float(

                probabilities[
                    index
                ]
            )


            st.write(

                f"**{class_name}**: "
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
            "❌ Prediction failed."
        )

        st.exception(
            error
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Research and educational use only. "
    "This application is not a medical diagnostic device."
)
