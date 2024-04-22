import streamlit as st
import tensorflow as tf
import PIL
from PIL import Image, ImageOps
import numpy as np

@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('91p11.h5')
    return model

model = load_model()

st.write("""
# Weather Prediction System
""")

file = st.file_uploader("Choose weather photo from computer", type=["jpg", "png"])

def import_and_predict(image_data, model):
    size = (32, 32)  # Match the input size with the Google Colab code
    image = ImageOps.fit(image_data, size, PIL.Image.LANCZOS)  # Use PIL.Image.LANCZOS for resizing
    img = np.asarray(image)
    img = img / 255.0  # Normalize pixel values
    img_reshape = img[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    class_names = ['Cloudy', 'Rain', 'Shine', 'Sunrise']
    string = "OUTPUT : " + class_names[np.argmax(prediction)]
    st.success(string)
