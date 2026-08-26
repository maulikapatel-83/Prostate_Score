# ============================================================
# PROSTATE CANCER GLEASON GRADING - STREAMLIT WEB APP
# CNN MODEL - 6 CLASS CLASSIFICATION
# ============================================================

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import gdown
import os

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Prostate Cancer Gleason Grading",
    page_icon="🔬",
    layout="centered"
)

# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = "RIL-SE-Clinic-model.keras"

IMAGE_SIZE = 224

NUM_CLASSES = 6

CLASS_NAMES = [
    "Gleason Grade 0",
    "Gleason Grade 1",
    "Gleason Grade 2",
    "Gleason Grade 3",
    "Gleason Grade 4",
    "Gleason Grade 5"
]

# ============================================================
# DOWNLOAD MODEL IF NOT EXISTS
# ============================================================

if not os.path.exists(MODEL_PATH):

    with st.spinner("Downloading trained CNN model..."):

        gdown.download(
            id="1Lj-IWr2Ghl76FCxfvUhah1UdZ0Ckbtfs",
            output=MODEL_PATH,
            quiet=False
        )

# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    return model


# ============================================================
# LOAD MODEL
# ============================================================

try:

    model = load_model()

except Exception as e:

    st.error("Unable to load the trained CNN model.")

    st.code(str(e))

    st.stop()


# ============================================================
# TITLE
# ============================================================

st.title("🔬 Prostate Cancer Gleason Grading")

st.subheader("Histopathological Image Classification")
st.divider()


# ============================================================
# IMAGE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "📤 Upload Histopathology Image",
    type=["jpg", "jpeg", "png"],
    help="Upload a JPG, JPEG, or PNG prostate histopathology image."
)


# ============================================================
# PREDICTION
# ============================================================

if uploaded_file is not None:

    # --------------------------------------------------------
    # READ IMAGE
    # --------------------------------------------------------

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    # --------------------------------------------------------
    # DISPLAY IMAGE
    # --------------------------------------------------------

    st.subheader("Uploaded Image")

    st.image(
        image,
        caption="Uploaded Histopathology Image",
        use_container_width=100
    )

    st.divider()

    # --------------------------------------------------------
    # PREDICT BUTTON
    # --------------------------------------------------------

    if st.button(
        "🔍 Predict Gleason Grade",
        use_container_width=True
    ):

        with st.spinner(
            "Analyzing histopathology image..."
        ):

            # Convert PIL image to NumPy
            image_array = np.array(image)

            # ------------------------------------------------
            # RESIZE
            # ------------------------------------------------

            image_array = cv2.resize(
                image_array,
                (IMAGE_SIZE, IMAGE_SIZE)
            )

            # ------------------------------------------------
            # IMPORTANT:
            # NO /255 NORMALIZATION
            # ------------------------------------------------

            image_array = image_array.astype(
                np.float32
            )

            # Add batch dimension
            image_array = np.expand_dims(
                image_array,
                axis=0
            )

            # ------------------------------------------------
            # MODEL PREDICTION
            # ------------------------------------------------

            predictions = model.predict(
                image_array,
                verbose=0
            )

            probabilities = predictions[0]

            # ------------------------------------------------
            # PREDICTED CLASS
            # ------------------------------------------------

            predicted_class = np.argmax(
                probabilities
            )

            # ------------------------------------------------
            # CONFIDENCE
            # ------------------------------------------------

            confidence = probabilities[
                predicted_class
            ]

        # ====================================================
        # DISPLAY RESULT
        # ====================================================

        st.success(
            "Prediction completed successfully!"
        )

        st.subheader(
            "Prediction Result"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Predicted Gleason Grade",
                CLASS_NAMES[predicted_class]
            )

        with col2:

            st.metric(
                "Confidence",
                f"{confidence * 100:.2f}%"
            )

        st.divider()

        # ====================================================
        # PROBABILITY DISTRIBUTION
        # ====================================================

        st.subheader(
            "Class Probability"
        )

        for i in range(NUM_CLASSES):

            probability = probabilities[i]

            st.write(
                f"**{CLASS_NAMES[i]}**"
            )

            st.progress(
                float(probability)
            )

            st.write(
                f"{probability * 100:.2f}%"
            )


# ============================================================
# NO IMAGE MESSAGE
# ============================================================

else:

    st.info(
        "👆 Please upload a prostate histopathology image "
        "to perform Gleason grading."
    )

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Prostate Cancer Gleason Grading Classification "
    "using Deep Learning with Reinforcement-Squeeze-and-Excitation."
)
