import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
from PIL import Image

# Load your pre-trained classification model
model = load_model('pest_classifier_model.h5')

# Define your classes
classes = ['ants', 'bees', 'beetle', 'caterpillar', 'earthworms', 'earwig', 'grasshopper', 'moth', 'slug', 'snail', 'wasp', 'weevil']

# Function to preprocess the image for EfficientNetB0
def preprocess_image(img):
    img = tf.convert_to_tensor(img, dtype=tf.float32)
    img = tf.image.resize(img, [224, 224])
    img = np.expand_dims(img, axis=0)
    return img

# Function to perform classification and overlay results on the frame
def classify_image(image):
    # Preprocess the image
    processed_image = preprocess_image(image)

    # Make prediction
    predictions = model.predict(processed_image)

    # Get the index of the predicted class
    predicted_class_index = np.argmax(predictions[0])

    # Get the class label
    predicted_class = classes[predicted_class_index]

    # Get the confidence score
    confidence = predictions[0][predicted_class_index]

    return predicted_class, confidence

# Function to classify and process video
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Classify the frame
        predicted_class, confidence = classify_image(frame_rgb)

        # Overlay predictions on the frame
        label_text = f"Class: {predicted_class} (Confidence: {confidence:.2f})"
        cv2.putText(frame, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Convert back to BGR for displaying with OpenCV
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        st.image(frame_bgr, channels="BGR", use_column_width=True)

    cap.release()

# Streamlit App
st.title("Pest Classification and Recommendation")

st.header("Upload an Image or a Video of a Pest")
uploaded_file = st.file_uploader("Choose an image or a video...", type=["jpg", "jpeg", "png", "mp4", "mov", "avi"])

if uploaded_file is not None:
    if uploaded_file.type.startswith('image'):
        # Handle image file
        image = Image.open(uploaded_file)
        image_np = np.array(image)

        # Classify the image
        predicted_class, confidence = classify_image(image_np)

        # Display the uploaded image
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Display the prediction
        st.write(f"Predicted Class: **{predicted_class}**")
        st.write(f"Confidence: **{confidence:.2f}**")

    elif uploaded_file.type.startswith('video'):
        # Handle video file
        video_path = f"temp_{uploaded_file.name}"
        with open(video_path, "wb") as f:
            f.write(uploaded_file.read())

        st.video(video_path)

        process_video(video_path)

        # Clean up by removing the temporary video file
        os.remove(video_path)

    # Optionally: Add a recommendation for pesticides (static for now)
    st.write("### Recommended Pesticide")
    st.write("For more details on suitable pesticides, consult your local agricultural extension service or pest control expert.")
