import streamlit as st
import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy import ndimage
from PIL import Image

def conversion(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image

def gaussian(image):
    blur = cv2.GaussianBlur(image, (5, 5), 0)
    return blur

def averagefilter(image):
    kernel = np.ones((5,5),np.float32)/25
    dst = cv2.filter2D(image, -1, kernel)
    return dst

def segmentation(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations = 2)

    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    labelarray, particle_count = ndimage.measurements.label(sure_bg)

    return sure_bg, particle_count

# Streamlit App
st.title("Pest Detection using Image Processing")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = np.array(Image.open(uploaded_file))

    st.subheader("Original Image")
    st.image(image, channels="BGR", use_column_width=True)

    # Conversion to Grayscale
    if st.button('Convert to Grayscale'):
        gray_image = conversion(image)
        st.subheader("Grayscale Image")
        st.image(gray_image, use_column_width=True)

    # Gaussian Blur
    if st.button('Apply Gaussian Blur'):
        gray_image = conversion(image)
        blur_image = gaussian(gray_image)
        st.subheader("Blurred Image")
        st.image(blur_image, use_column_width=True)

    # Apply Average Filter
    if st.button('Apply Average Filter'):
        gray_image = conversion(image)
        blur_image = gaussian(gray_image)
        averaged_image = averagefilter(blur_image)
        st.subheader("Averaged Image")
        st.image(averaged_image, use_column_width=True)

    # Segmentation
    if st.button('Segment and Count Pests'):
        gray_image = conversion(image)
        blur_image = gaussian(gray_image)
        averaged_image = averagefilter(blur_image)
        segmented_image, pest_count = segmentation(averaged_image)
        st.subheader("Segmented Image")
        st.image(segmented_image, use_column_width=True)
        st.write(f"Number of pests detected: {pest_count}")
