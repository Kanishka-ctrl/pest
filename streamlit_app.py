import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from PIL import Image
import numpy as np

@st.cache(allow_output_mutation=True)
def load_pretrained_model():
    model = MobileNetV2(weights='imagenet')
    return model

def preprocess_image(image, target_size=(224, 224)):
    image = image.resize(target_size)
    image = np.array(image)
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def predict_image(image, model):
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    return decode_predictions(predictions, top=3)

# Streamlit app interface
st.title("Pest Classification using MobileNetV2")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    model = load_pretrained_model()
    predictions = predict_image(image, model)

    st.subheader("Predictions")
    for i, (imagenet_id, label, score) in enumerate(predictions[0]):
        st.write(f"{i+1}: {label} ({score:.4f})")
