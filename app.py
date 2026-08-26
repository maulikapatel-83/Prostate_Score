# ============================================================
# PROSTATE CANCER GLEASON GRADING - STREAMLIT WEB APP
# ============================================================

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Prostate Cancer Gleason Grading",
    page_icon="🔬",
    layout="centered"
)

# ============================================================
# CUSTOM SQUEEZE-AND-EXCITATION LAYER
# ============================================================

class SqueezeAndExcitation(tf.keras.layers.Layer):

    def __init__(self, reduction_ratio=16, **kwargs):
        super(SqueezeAndExcitation, self).__init__(**kwargs)
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):

        self.global_avg_pool = tf.keras.layers.GlobalAveragePooling2D()

        self.dense1 = tf.keras.layers.Dense(
            input_shape[-1] // self.reduction_ratio,
            activation="relu",
            use_bias=False
        )

        self.dense2 = tf.keras.layers.Dense(
            input_shape[-1],
            activation="sigmoid",
            use_bias=False
        )

        super().build(input_shape)

    def call(self, inputs):

        # Squeeze
        squeeze = self.global_avg_pool(inputs)

        # Reshape
        squeeze = tf.reshape(
            squeeze,
            (-1, 1, 1, tf.shape(squeeze)[-1])
        )

        # Excitation
        excitation = self.dense1(squeeze)
        excitation = self.dense2(excitation)

        # Scale
        return inputs * excitation

    def get_config(self):

        config = super().get_config()

        config.update({
            "reduction_ratio": self.reduction_ratio
        })

        return config


# ============================================================
# SETTINGS
# ============================================================
import gdown
import os

MODEL_PATH = "RIL-SE-Clinic-model.keras"

if not os.path.exists(MODEL_PATH):
    gdown.download( id= "144SOJTBGV2s74XXa064p8QqSEbUJaQHQ", output="RIL-SE-Clinic-model.keras", quiet=False)
    #https://drive.google.com/file/d/144SOJTBGV2s74XXa064p8QqSEbUJaQHQ/view?usp=sharing
IMAGE_SIZE = 224
NUM_CLASSES = 6
# Change these names according to your dataset
CLASS_NAMES = [
    "Gleason Grade 0",
    "Gleason Grade 1",
    "Gleason Grade 2",
    "Gleason Grade 3",
    "Gleason Grade 4",
    "Gleason Grade 5"
]


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={
            "SqueezeAndExcitation": SqueezeAndExcitation
        }
    )

    return model


# ============================================================
# LOAD MODEL
# ============================================================

try:

    model = load_model()

except Exception as e:

    st.error("Unable to load the trained model.")

    st.code(str(e))

    st.stop()


# ============================================================
# TITLE
# ============================================================

st.title("🔬 Prostate Cancer Gleason Grading")

st.subheader("Histopathological Image Classification")

st.write(
    """
    Upload a prostate histopathology image to obtain the
    predicted Gleason grade from the trained deep learning model.
    """
)

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

    # Read image
    image = Image.open(uploaded_file).convert("RGB")

    # Display image
    st.subheader("Uploaded Image")

    st.image(
        image,
        caption="Uploaded Histopathology Image",
        use_container_width=True
    )

    st.divider()

    # Predict button
    if st.button(
        "🔍 Predict Gleason Grade",
        use_container_width=True
    ):

        with st.spinner("Analyzing histopathology image..."):

            # Resize image
            img = image.resize(
                (IMAGE_SIZE, IMAGE_SIZE)
            )

            # Convert to NumPy
            img_array = np.array(img)

            # Add batch dimension
            img_array = np.expand_dims(
                img_array,
                axis=0
            )

            # Prediction
            predictions = model.predict(
                img_array,
                verbose=0
            )

            probabilities = predictions[0]

            # Predicted class
            predicted_class = np.argmax(
                probabilities
            )

            # Confidence
            confidence = probabilities[
                predicted_class
            ]


        # ====================================================
        # DISPLAY RESULT
        # ====================================================

        st.success("Prediction completed successfully!")

        st.subheader("Prediction Result")

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

        st.subheader("Class Probability")

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


        # ====================================================
        # RAW PREDICTION
        # ====================================================

        st.divider()

        st.subheader("Model Output")

        prediction_data = {
            CLASS_NAMES[i]: f"{probabilities[i] * 100:.2f}%"
            for i in range(NUM_CLASSES)
        }

        st.table(prediction_data)


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
    "using Deep Learning with Squeeze-and-Excitation."
)

st.caption(
    "⚠️ Research/educational use only. "
    "This application is not intended for clinical diagnosis."
)
