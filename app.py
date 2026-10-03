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
# TITLE
# ============================================================

st.title("🦴 Bone Fracture Detection")

st.write(
    "Upload a bone X-ray image to classify it as "
    "**fractured** or **not fractured**."
)


# ============================================================
# REQUIRED FILE CHECK
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
        "Upload the missing files to the same "
        "GitHub folder as app.py."
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
        "Configuration loading failed."
    )

    st.exception(error)

    st.stop()


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
        str(MODEL_PATH),

        custom_objects={
            "preprocess_input":
                preprocess_input
        },

        compile=False,

        safe_mode=False
    )

    return model


try:

    model = load_model()

except Exception as error:

    st.error(
        "Model loading failed."
    )

    st.exception(error)

    st.stop()


# ============================================================
# MODEL STATUS
# ============================================================

st.success(
    "✅ DenseNet121 model loaded successfully."
)


# ============================================================
# GET CONFIG
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
        [224, 224]
    )
)


# ============================================================
# UPLOAD IMAGE
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
        # BATCH DIMENSION
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
        # PROBABILITIES
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
