import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Model load karein
model = tf.keras.models.load_model('mnist_model.h5')

st.title("Handwritten Digit Classifier")
st.write("Apni image upload karein ya yahan draw karein.")

# Image upload ka option
file = st.file_sidebar.file_uploader("Upload an image (28x28)", type=["jpg", "png"])

if file:
    img = Image.open(file).convert('L').resize((28,28))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 784)
    
    prediction = model.predict(img_array)
    st.image(img, caption="Uploaded Image", width=150)
    st.write(f"Mujhe lagta hai ye **{np.argmax(prediction)}** hai!")
