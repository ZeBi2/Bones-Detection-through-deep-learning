"""
Bone Fracture Detection — Streamlit Application
------------------------------------------------
DenseNet121 binary image classifier:
    0 -> fractured
    1 -> not_fractured

The application uses the same deployment metadata produced by the
training notebook. The saved model contains DenseNet preprocessing,
so uploaded RGB images are passed to the model without manually
applying DenseNet preprocess_input a second time.

Run:
    streamlit run app.py
"""

from pathlib import Path
import pickle
import json
import io

import numpy as np
import streamlit as st
from PIL import Image

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.densenet import preprocess_input


# ============================================================
# 1. APPLICATION CONFIGURATION
# ============================================================

APP_DIR = Path(__file__).resolve().parent

MODEL_PATH = APP_DIR / "best_bones_fracture_model.keras"
MODEL_CONFIG_PATH = APP_DIR / "model_config.pkl"
CLASS_MAPPING_PATH = APP_DIR / "class_mapping.pkl"
IMAGE_CONFIG_PATH = APP_DIR / "image_config.pkl"
METADATA_PATH = APP_DIR / "best_model_metadata.json"

APP_TITLE = "Bone Fracture Detection"
APP_ICON = "🦴"
IMAGE_SIZE = (224, 224)
SUPPORTED_TYPES = ["jpg", "jpeg", "png", "bmp", "tif", "tiff"]


# ============================================================
# 2. PAGE DESIGN
# ============================================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main {
        background: linear-gradient(180deg, #f8fafc 0%, #ffffff 42%);
    }

    .hero {
        padding: 1.8rem 2rem;
        border-radius: 22px;
        margin-bottom: 1.4rem;
        background: linear-gradient(135deg, #0f172a, #1e293b);
        color: white;
        box-shadow: 0 12px 30px rgba(15, 23, 42, 0.14);
    }

    .hero h1 {
        margin: 0 0 .35rem 0;
        font-size: 2.35rem;
        letter-spacing: -0.04em;
    }

    .hero p {
        margin: 0;
        color: #cbd5e1;
        font-size: 1.02rem;
    }

    .metric-card {
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 1rem 1.1rem;
        background: white;
        box-shadow: 0 5px 18px rgba(15, 23, 42, 0.05);
    }

    .metric-label {
        color: #64748b;
        font-size: .84rem;
        margin-bottom: .25rem;
    }

    .metric-value {
        color: #0f172a;
        font-size: 1.15rem;
        font-weight: 700;
    }

    .result-box {
        border-radius: 18px;
        padding: 1.25rem;
        border: 1px solid #e2e8f0;
        background: white;
        margin-top: .75rem;
    }

    .result-title {
        font-size: 1.45rem;
        font-weight: 800;
        margin-bottom: .25rem;
    }

    .result-subtitle {
        color: #64748b;
        margin-bottom: .8rem;
    }

    .small-note {
        color: #64748b;
        font-size: .85rem;
        line-height: 1.5;
    }

    div[data-testid="stFileUploader"] {
        border-radius: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 3. SAFE FILE LOADING
# ============================================================

def load_pickle(path: Path):
    """Load a pickle file and return its Python object."""
    with path.open("rb") as file:
        return pickle.load(file)


@st.cache_resource(show_spinner="Loading DenseNet121 model...")
def load_app_model():
    """
    Load the saved Keras model.

    preprocess_input is registered because the training notebook
    embeds DenseNet preprocessing in a Lambda/preprocessing layer.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH.name}"
        )

    return load_model(
        MODEL_PATH,
        custom_objects={"preprocess_input": preprocess_input},
        compile=False,
        safe_mode=False,
    )


@st.cache_data
def load_metadata():
    """Load deployment metadata generated during model export."""
    configs = {}

    for path, key in [
        (MODEL_CONFIG_PATH, "model_config"),
        (CLASS_MAPPING_PATH, "class_mapping"),
        (IMAGE_CONFIG_PATH, "image_config"),
    ]:
        if path.exists():
            configs[key] = load_pickle(path)

    if METADATA_PATH.exists():
        with METADATA_PATH.open("r", encoding="utf-8") as file:
            configs["metadata"] = json.load(file)

    return configs


# ============================================================
# 4. IMAGE PREPARATION
# ============================================================

def prepare_image(uploaded_file):
    """
    Convert an uploaded image to RGB and resize it to 224x224.

    The returned tensor is float32 in RGB order. DenseNet
    preprocessing is handled by the saved model itself.
    """
    image = Image.open(uploaded_file).convert("RGB")
    display_image = image.copy()

    resized = image.resize(IMAGE_SIZE)
    array = np.asarray(resized, dtype=np.float32)

    # Batch dimension: (224, 224, 3) -> (1, 224, 224, 3)
    tensor = np.expand_dims(array, axis=0)

    return display_image, tensor


# ============================================================
# 5. PREDICTION
# ============================================================

def predict_image(model, tensor, class_mapping):
    """
    Run inference and return prediction details.

    The saved model has two outputs. The application supports
    both a probability vector output and a single sigmoid output.
    """
    raw = model.predict(tensor, verbose=0)
    output = np.asarray(raw)

    if output.ndim == 2 and output.shape[1] == 2:
        probabilities = output[0].astype(float)
        predicted_index = int(np.argmax(probabilities))
    else:
        # Fallback for a single sigmoid output.
        p_fractured = float(output.reshape(-1)[0])
        probabilities = np.array(
            [p_fractured, 1.0 - p_fractured],
            dtype=float,
        )
        predicted_index = int(np.argmax(probabilities))

    index_to_class = {
        int(index): name for name, index in class_mapping.items()
    }

    predicted_class = index_to_class.get(
        predicted_index,
        "fractured" if predicted_index == 0 else "not_fractured",
    )

    confidence = float(probabilities[predicted_index])

    return {
        "class": predicted_class,
        "index": predicted_index,
        "confidence": confidence,
        "probabilities": probabilities,
    }


# ============================================================
# 6. EXPLANATION / GRAD-CAM
# ============================================================

def find_last_conv_layer(model):
    """Find the last Conv2D-like layer for Grad-CAM."""
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name

        # DenseNet can contain nested models.
        if hasattr(layer, "layers"):
            for nested in reversed(layer.layers):
                if isinstance(nested, tf.keras.layers.Conv2D):
                    return nested.name

    return None


def make_gradcam(model, tensor, class_index):
    """
    Generate a Grad-CAM heatmap when the architecture exposes
    a usable convolutional layer.

    This is intentionally optional: prediction still works if
    Grad-CAM is unavailable for a particular serialized model.
    """
    layer_name = find_last_conv_layer(model)
    if layer_name is None:
        return None

    try:
        conv_layer = model.get_layer(layer_name)

        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[conv_layer.output, model.output],
        )

        image_tensor = tf.cast(tensor, tf.float32)

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(image_tensor)

            if predictions.shape[-1] == 2:
                target = predictions[:, class_index]
            else:
                target = predictions[:, 0]

        gradients = tape.gradient(target, conv_outputs)

        if gradients is None:
            return None

        pooled_gradients = tf.reduce_mean(
            gradients,
            axis=(1, 2),
        )

        conv_outputs = conv_outputs[0]
        pooled_gradients = pooled_gradients[0]

        heatmap = tf.reduce_sum(
            conv_outputs * pooled_gradients,
            axis=-1,
        )

        heatmap = tf.maximum(heatmap, 0)
        max_value = tf.reduce_max(heatmap)

        heatmap = tf.where(
            max_value > 0,
            heatmap / max_value,
            heatmap,
        )

        return heatmap.numpy()

    except Exception:
        return None


def overlay_heatmap(image, heatmap, alpha=0.42):
    """Create a visual Grad-CAM overlay without requiring OpenCV."""
    import matplotlib.pyplot as plt

    heatmap_image = Image.fromarray(
        np.uint8(heatmap * 255)
    ).resize(image.size)

    heatmap_array = np.asarray(heatmap_image)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.imshow(image)
    ax.imshow(
        heatmap_array,
        cmap="jet",
        alpha=alpha,
        interpolation="bilinear",
    )
    ax.axis("off")
    fig.tight_layout(pad=0)

    buffer = io.BytesIO()
    fig.savefig(
        buffer,
        format="png",
        dpi=160,
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close(fig)

    buffer.seek(0)
    return Image.open(buffer).copy()


# ============================================================
# 7. SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## 🦴 Bone Fracture AI")
    st.caption("DenseNet121 • Streamlit")

    st.divider()

    st.markdown("### Model")
    st.write("**Architecture:** DenseNet121")
    st.write("**Input:** 224 × 224 × 3")
    st.write("**Classes:** 2")

    st.divider()

    st.markdown("### Classes")
    st.write("🦴 **Fractured**")
    st.write("✓ **Not fractured**")

    st.divider()

    st.markdown("### Explanation")
    show_gradcam = st.checkbox(
        "Generate Grad-CAM",
        value=True,
        help="Optional visual explanation of the model prediction.",
    )

    st.divider()

    st.markdown(
        """
        <div class="small-note">
        <b>Research / educational use only.</b><br>
        This application is not a medical diagnostic device and
        should not replace assessment by a qualified healthcare
        professional.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# 8. HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>🦴 Bone Fracture Detection</h1>
        <p>
            Upload an X-ray image and let the trained DenseNet121
            model classify it as fractured or not fractured.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 9. MODEL STATUS
# ============================================================

try:
    metadata = load_metadata()
    model_config = metadata.get("model_config", {})
    class_mapping = metadata.get(
        "class_mapping",
        {"fractured": 0, "not_fractured": 1},
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">Architecture</div>
                <div class="metric-value">DenseNet121</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">Input Size</div>
                <div class="metric-value">224 × 224</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">Classes</div>
                <div class="metric-value">2</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        params = model_config.get("parameters", 7_300_418)
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Parameters</div>
                <div class="metric-value">{params:,}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

except Exception as exc:
    st.error(f"Deployment metadata could not be loaded: {exc}")
    st.stop()


# ============================================================
# 10. UPLOAD AREA
# ============================================================

st.subheader("Upload an X-ray image")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=SUPPORTED_TYPES,
    help="Supported: JPG, JPEG, PNG, BMP, TIF, TIFF",
)

if uploaded_file is None:
    st.info(
        "Upload an image to start prediction. "
        "The model expects RGB input resized to 224 × 224 pixels."
    )
    st.stop()


# ============================================================
# 11. PREVIEW
# ============================================================

try:
    original_image, tensor = prepare_image(uploaded_file)
except Exception as exc:
    st.error(f"Could not read this image: {exc}")
    st.stop()

preview_col, details_col = st.columns([1.25, 1])

with preview_col:
    st.image(
        original_image,
        caption="Uploaded X-ray",
        use_container_width=True,
    )

with details_col:
    st.markdown("### Image details")
    st.write(f"**Filename:** {uploaded_file.name}")
    st.write(f"**Original size:** {original_image.size[0]} × {original_image.size[1]}")
    st.write("**Color:** RGB")
    st.write("**Model size:** 224 × 224")


# ============================================================
# 12. PREDICT BUTTON
# ============================================================

st.divider()

predict_col, reset_col = st.columns([3, 1])

with predict_col:
    run_prediction = st.button(
        "🔍 Analyze X-ray",
        type="primary",
        use_container_width=True,
    )

with reset_col:
    if st.button("Clear", use_container_width=True):
        st.rerun()


if not run_prediction:
    st.stop()


# ============================================================
# 13. RUN MODEL
# ============================================================

try:
    model = load_app_model()

    with st.spinner("Analyzing image..."):
        result = predict_image(
            model,
            tensor,
            class_mapping,
        )

except Exception as exc:
    st.error(
        "Model inference failed. "
        "Check that best_bones_fracture_model.keras is present "
        "and compatible with the TensorFlow version in requirements.txt."
    )
    st.exception(exc)
    st.stop()


# ============================================================
# 14. RESULT
# ============================================================

predicted_class = result["class"]
confidence = result["confidence"]
probabilities = result["probabilities"]

if predicted_class == "fractured":
    display_name = "Fracture Detected"
    icon = "⚠️"
else:
    display_name = "No Fracture Detected"
    icon = "✅"

st.markdown(
    f"""
    <div class="result-box">
        <div class="result-title">{icon} {display_name}</div>
        <div class="result-subtitle">
            Model prediction: <b>{predicted_class}</b>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

metric1, metric2, metric3 = st.columns(3)

with metric1:
    st.metric(
        "Prediction",
        predicted_class.replace("_", " ").title(),
    )

with metric2:
    st.metric(
        "Confidence",
        f"{confidence * 100:.2f}%",
    )

with metric3:
    st.metric(
        "Predicted Index",
        str(result["index"]),
    )


# ============================================================
# 15. CLASS PROBABILITIES
# ============================================================

st.subheader("Prediction probabilities")

probability_col1, probability_col2 = st.columns(2)

fractured_probability = float(probabilities[0])
normal_probability = float(probabilities[1])

with probability_col1:
    st.write("**Fractured**")
    st.progress(fractured_probability)
    st.caption(f"{fractured_probability * 100:.2f}%")

with probability_col2:
    st.write("**Not fractured**")
    st.progress(normal_probability)
    st.caption(f"{normal_probability * 100:.2f}%")


# ============================================================
# 16. GRAD-CAM
# ============================================================

if show_gradcam:
    st.divider()
    st.subheader("🔬 Visual Explanation — Grad-CAM")

    with st.spinner("Generating Grad-CAM explanation..."):
        heatmap = make_gradcam(
            model,
            tensor,
            result["index"],
        )

    if heatmap is None:
        st.warning(
            "Grad-CAM could not be generated for this serialized model. "
            "The prediction itself completed successfully."
        )
    else:
        overlay = overlay_heatmap(
            original_image,
            heatmap,
        )

        xai_col1, xai_col2 = st.columns(2)

        with xai_col1:
            st.image(
                original_image,
                caption="Original image",
                use_container_width=True,
            )

        with xai_col2:
            st.image(
                overlay,
                caption="Grad-CAM attention map",
                use_container_width=True,
            )

        st.caption(
            "Grad-CAM highlights image regions that contributed "
            "to the selected CNN prediction. It should be treated "
            "as an explanation aid, not as a clinical finding."
        )


# ============================================================
# 17. FINAL DISCLAIMER
# ============================================================

st.divider()

st.warning(
    "Research and educational use only. This application is not "
    "a medical diagnostic device and must not be used as a substitute "
    "for evaluation by a qualified healthcare professional."
)
