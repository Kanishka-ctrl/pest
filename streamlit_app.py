import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import EfficientNetV2B0  # Import the specific EfficientNetV2 model
from PIL import Image
import numpy as np

# Load pre-trained model with weights from ImageNet or any specific dataset
@st.cache(allow_output_mutation=True)
def load_pretrained_model():
    # Assuming you downloaded v2b0-21k.h5 to the same directory
    model = load_model('v2b0-21k.h5')  # Replace with the actual path to your model
    return model

# Preprocess the image for the specific model input
def preprocess_image(image, model_input_size):
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = tf.image.resize(image, model_input_size)
    image = tf.keras.applications.efficientnet_v2.preprocess_input(image)  # Preprocessing specific to EfficientNetV2
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Perform prediction using the model
def predict_image(image, model):
    processed_image = preprocess_image(image, (224, 224))  # Example size, adjust based on model
    predictions = model.predict(processed_image)
    return predictions

# Streamlit app interface
st.title("Pest Classification using EfficientNetV2")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_np = np.array(image)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    model = load_pretrained_model()
    predictions = predict_image(image_np, model)

    st.subheader("Predicted Class")
    predicted_class = np.argmax(predictions, axis=1)
    st.write(f"Class ID: {predicted_class[0]}")

    st.subheader("Prediction Confidence")
    st.write(predictions)
