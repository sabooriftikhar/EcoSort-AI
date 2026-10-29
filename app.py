import json
import os

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(
    page_title="EcoSort AI",
    page_icon="♻️",
    layout="wide"
)

MODEL_PATH = "model.keras"
CLASSES_PATH = "class_names.json"

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_data
def load_classes():
    with open(CLASSES_PATH, "r") as f:
        return json.load(f)

model = load_model()
class_names = load_classes()

st.title("♻️ EcoSort AI")
st.subheader("Deep Learning Waste Classification System")

st.write(
    "Upload a waste image and our trained CNN classifier "
    "will predict its category."
)

st.info(
    "This is an academic image-classification demo. "
    "Predictions may be incorrect, especially for unfamiliar images."
)

uploaded_file = st.file_uploader(
    "Upload a waste image",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    left, right = st.columns(2)

    with left:
        st.image(image, caption="Uploaded image", use_container_width=True)

    with st.spinner("Classifying image..."):
        resized = image.resize((160, 160))
        array = np.asarray(resized, dtype=np.float32)
        array = np.expand_dims(array, axis=0)

        probabilities = model.predict(array, verbose=0)[0]

    predicted_index = int(np.argmax(probabilities))
    predicted_class = class_names[predicted_index]
    confidence = float(probabilities[predicted_index]) * 100

    with right:
        st.markdown("### Prediction")
        st.success(predicted_class.replace("-", " ").title())
        st.metric("Model confidence", f"{confidence:.2f}%")

        st.caption(
            "Confidence is the model's score, not a guarantee "
            "that the prediction is correct."
        )

    st.markdown("### Top 3 predictions")

    top_indices = np.argsort(probabilities)[::-1][:3]

    for index in top_indices:
        st.write(
            f"**{class_names[index].replace('-', ' ').title()}**"
        )
        st.progress(float(probabilities[index]))

st.divider()

st.header("About the project")

st.write(
    "EcoSort AI uses transfer learning with MobileNetV2 "
    "to classify images into 12 waste-related categories. "
    "The model was trained on the Kaggle Garbage Classification dataset."
)

st.header("Training results")

if os.path.exists("metrics.json"):
    with open("metrics.json", "r") as f:
        metrics = json.load(f)

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Training accuracy",
        f"{metrics['final_training_accuracy'] * 100:.2f}%"
    )

    col2.metric(
        "Validation accuracy",
        f"{metrics['final_validation_accuracy'] * 100:.2f}%"
    )

    col3.metric("Epochs completed", metrics["epochs_completed"])

for filename, title in [
    ("accuracy.png", "Accuracy Graph"),
    ("loss.png", "Loss Graph"),
    ("confusion_matrix.png", "Confusion Matrix")
]:
    if os.path.exists(filename):
        st.subheader(title)
        st.image(filename, use_container_width=True)

st.header("Project resources")

st.markdown(
    "- Dataset: [Kaggle Garbage Classification]"
    "(https://www.kaggle.com/datasets/mostafaabla/garbage-classification)\n"
    "- Source code and model weights: Add your GitHub repository link in README.\n"
    "- Project report: Add your report PDF link here.\n"
    "- Presentation and video: Add your presentation and YouTube links here.\n"
    "- LinkedIn: Add your project post link here."
)

st.caption("EcoSort AI | Deep Learning Semester Project")