from pathlib import Path
import json

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "clothing_mobilenetv2_final.keras"
CLASS_NAMES_PATH = BASE_DIR / "config" / "class_names.json"

IMAGE_SIZE = (180, 180)
TOP_K = 3


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Clothing Classification AI",
    page_icon="👕",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# PROFESSIONAL CSS
# IMPORTANT:
# We use st.html() instead of st.markdown() for CSS.
# ============================================================

st.html(
    """
    <style>

    /* ======================================================
       PAGE
       ====================================================== */

    html, body {
        margin: 0;
        padding: 0;
    }

    [data-testid="stAppViewContainer"] {
        background: #f4f6fa;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stToolbar"] {
        display: none;
    }

    .main .block-container {
        max-width: 1180px;

        padding-top: 18px;
        padding-bottom: 12px;
        padding-left: 20px;
        padding-right: 20px;
    }


    /* ======================================================
       REMOVE EXCESS STREAMLIT SPACING
       ====================================================== */

    div[data-testid="stVerticalBlock"] {
        gap: 0.45rem;
    }

    div[data-testid="stHorizontalBlock"] {
        gap: 1rem;
    }


    /* ======================================================
       HEADINGS
       ====================================================== */

    h1 {
        font-size: 30px !important;
        font-weight: 800 !important;
        text-align: center;
        color: #111827 !important;
        margin-bottom: 2px !important;
    }

    .subtitle {
        text-align: center;
        color: #6b7280;
        font-size: 13px;
        margin-bottom: 14px;
    }


    /* ======================================================
       COLUMN CARDS
       ====================================================== */

    [data-testid="column"] {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 16px;

        box-shadow:
            0 5px 18px rgba(15, 23, 42, 0.06);
    }


    /* ======================================================
       FILE UPLOADER
       ====================================================== */

    [data-testid="stFileUploader"] {
        background: #f8fafc;
        border: 1px dashed #cbd5e1;
        border-radius: 10px;
        padding: 4px;
    }


    /* ======================================================
       UPLOADED IMAGE
       ====================================================== */

    [data-testid="stImage"] img {
        max-height: 260px;
        width: auto;
        object-fit: contain;
        border-radius: 12px;
    }


    /* ======================================================
       BUTTON
       ====================================================== */

    .stButton > button {
        width: 100%;
        height: 44px;

        border: none;
        border-radius: 10px;

        background: #2563eb;
        color: white;

        font-size: 14px;
        font-weight: 700;

        transition: all 0.15s ease;
    }

    .stButton > button:hover {
        background: #1d4ed8;
        color: white;
    }


    /* ======================================================
       METRIC
       ====================================================== */

    [data-testid="stMetric"] {
        background: #f8fafc;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 10px;
    }


    /* ======================================================
       PROGRESS BAR
       ====================================================== */

    [data-testid="stProgressBar"] {
        margin-top: -4px;
        margin-bottom: 4px;
    }


    /* ======================================================
       INFO / SUCCESS
       ====================================================== */

    [data-testid="stAlert"] {
        border-radius: 10px;
    }


    /* ======================================================
       FOOTER INFO
       ====================================================== */

    .model-info {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 10px;

        padding: 8px 12px;

        text-align: center;

        color: #6b7280;
        font-size: 11px;

        margin-top: 10px;
    }


    /* ======================================================
       MOBILE
       ====================================================== */

    @media (max-width: 800px) {

        .main .block-container {
            padding-left: 12px;
            padding-right: 12px;
        }

        h1 {
            font-size: 24px !important;
        }

    }

    </style>
    """
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found:\n{MODEL_PATH}"
        )

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    return model


# ============================================================
# LOAD CLASS NAMES
# ============================================================

@st.cache_data
def load_class_names():

    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(
            f"Class names file not found:\n{CLASS_NAMES_PATH}"
        )

    with open(
        CLASS_NAMES_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        class_names = json.load(file)

    return class_names


# ============================================================
# PREPROCESS IMAGE
# ============================================================

def preprocess_image(image):

    image = image.convert("RGB")

    image = image.resize(IMAGE_SIZE)

    image_array = np.asarray(
        image,
        dtype=np.float32
    )

    image_array = image_array / 255.0

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# ============================================================
# PREDICTION
# ============================================================

def predict_image(
    model,
    image,
    class_names
):

    processed_image = preprocess_image(
        image
    )

    predictions = model.predict(
        processed_image,
        verbose=0
    )[0]

    top_indices = np.argsort(
        predictions
    )[::-1][:TOP_K]

    results = []

    for index in top_indices:

        results.append(
            {
                "class": class_names[index],
                "confidence": float(
                    predictions[index]
                )
            }
        )

    return results


# ============================================================
# LOAD MODEL + CLASSES
# ============================================================

try:

    model = load_model()

    class_names = load_class_names()

except Exception as error:

    st.error(
        f"Unable to load application resources.\n\n{error}"
    )

    st.stop()


# ============================================================
# HEADER
# ============================================================

st.title("👕 Clothing Classification AI")

st.markdown(
    '<div class="subtitle">'
    'AI-powered clothing image classification using MobileNetV2'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# MAIN TWO-COLUMN LAYOUT
# ============================================================

left_column, right_column = st.columns(
    [1, 1],
    gap="large"
)


# ============================================================
# LEFT COLUMN
# ============================================================

with left_column:

    st.subheader("📷 Upload Clothing Image")

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ],
        label_visibility="collapsed"
    )

    image = None

    if uploaded_file is not None:

        try:

            image = Image.open(
                uploaded_file
            ).convert("RGB")

            st.image(
                image,
                width=260
            )

            st.caption(
                f"{uploaded_file.name}  •  "
                f"{image.width} × {image.height}px"
            )

        except Exception:

            st.error(
                "Unable to read this image."
            )

            image = None

    else:

        st.info(
            "👕 Upload a clothing image to begin."
        )

    predict_clicked = st.button(
        "🔍 Predict Clothing",
        type="primary",
        use_container_width=True
    )


# ============================================================
# RIGHT COLUMN
# ============================================================

with right_column:

    st.subheader("🎯 Prediction Results")

    if image is None:

        st.info(
            "Prediction results will appear here."
        )

        st.caption(
            "Upload an image and click "
            "**Predict Clothing**."
        )

    elif predict_clicked:

        with st.spinner(
            "Analyzing clothing image..."
        ):

            results = predict_image(
                model,
                image,
                class_names
            )

        best_result = results[0]

        predicted_class = best_result["class"]

        confidence = best_result["confidence"]

        # ----------------------------------------------------
        # MAIN PREDICTION
        # ----------------------------------------------------

        st.success(
            f"Prediction: **{predicted_class}**"
        )

        st.metric(
            label="Model Confidence",
            value=f"{confidence * 100:.2f}%"
        )

        # ----------------------------------------------------
        # TOP 3
        # ----------------------------------------------------

        st.markdown("##### Top 3 Predictions")

        for position, result in enumerate(
            results,
            start=1
        ):

            class_name = result["class"]

            probability = result["confidence"]

            percentage = probability * 100

            col_name, col_probability = st.columns(
                [3, 1]
            )

            with col_name:

                st.write(
                    f"**{position}. {class_name}**"
                )

            with col_probability:

                st.write(
                    f"{percentage:.2f}%"
                )

            st.progress(
                probability
            )

    else:

        st.info(
            "🚀 Ready to classify."
        )

        st.caption(
            "Click **Predict Clothing** to run the model."
        )


# ============================================================
# MODEL INFORMATION
# ============================================================

st.markdown(
    '<div class="model-info">'
    '<b>Model:</b> MobileNetV2'
    ' &nbsp;•&nbsp; '
    '<b>Classes:</b> 20'
    ' &nbsp;•&nbsp; '
    '<b>Input:</b> 180 × 180'
    ' &nbsp;•&nbsp; '
    '<b>Framework:</b> TensorFlow / Keras'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SUPPORTED CLASSES
# ============================================================

with st.expander(
    "View Supported Clothing Classes"
):

    class_columns = st.columns(4)

    for index, class_name in enumerate(
        class_names
    ):

        with class_columns[
            index % 4
        ]:

            st.caption(
                f"• {class_name}"
            )