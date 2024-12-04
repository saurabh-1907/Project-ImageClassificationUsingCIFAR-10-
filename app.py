# app.py

import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array
from PIL import Image
import numpy as np

# Load the trained model
model = load_model("classify.keras")

# Class names for CIFAR-10
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Streamlit app
st.title("Image Classification with CNN (CIFAR-10)")
st.write("This is a simple image classification web app.")
st.write("***The model predicts one of these classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, or truck.***")

uploaded_file = st.file_uploader("Upload an image (32x32) to classify", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    img_array = img_to_array(image.resize((32, 32))) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make a prediction
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]

    # Display the prediction
    st.write(f"Predicted Class: **{predicted_class}**")
