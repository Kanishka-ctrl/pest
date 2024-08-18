import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
from PIL import Image
import os

# Define your classes
classes = ['ants', 'bees', 'beetle', 'caterpillar', 'earthworms', 'earwig', 'grasshopper', 'moth', 'slug', 'snail', 'wasp', 'weevil']

# Custom layer to handle potential issues with DepthwiseConv2D
class CustomDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
    def __init__(self, **kwargs):
        kwargs.pop('groups', None)  # Remove 'groups' argument if present
        super(CustomDepthwiseConv2D, self).__init__(**kwargs)

# Load your pre-trained classification model
try:
    custom_objects = {'DepthwiseConv2D': CustomDepthwiseConv2D}
    model = load_model('pest_classifier_model.h5', custom_objects=custom_objects)
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")

# Function to preprocess the image
def preprocess_image(img):
    img = tf.convert_to_tensor(img, dtype=tf.float32)
    img = tf.image.resize(img, [224, 224])
    img = np.expand_dims(img, axis=0)
    return img

# Function to perform classification
def classify_image(image):
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    predicted_class_index = np.argmax(predictions[0])
    predicted_class = classes[predicted_class_index]
    confidence = predictions[0][predicted_class_index]
    return predicted_class, confidence

# Streamlit App
st.title("Pest Classification and Recommendation")

st.header("Upload an Image or a Video of a Pest")
uploaded_file = st.file_uploader("Choose an image or a video...", type=["jpg", "jpeg", "png", "mp4", "mov", "avi"])

if uploaded_file is not None:
    if uploaded_file.type.startswith('image'):
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        predicted_class, confidence = classify_image(image_np)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write(f"Predicted Class: **{predicted_class}**")
        st.write(f"Confidence: **{confidence:.2f}**")

    elif uploaded_file.type.startswith('video'):
        video_path = f"temp_{uploaded_file.name}"
        with open(video_path, "wb") as f:
            f.write(uploaded_file.read())
        st.video(video_path)
        st.write("Video classification is currently not supported.")
        os.remove(video_path)

# Optionally: Add a recommendation for pesticides (static for now)
st.write("### Recommended Pesticide")
st.write("For more details on suitable pesticides, consult your local agricultural extension service or pest control expert.")
