from matplotlib.path import Path
import streamlit as st
import pandas as pd
from PIL import Image
import tensorflow as tf
import numpy as np
from pathlib import Path

"""
Deep Classifier Project.


Image Clasification: CAT and DOG
"""

model = tf.keras.models.load_model(Path("artifact/training/model.h5"))
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    
    image = Image.open(uploaded_file)
    img= image.resize((224,224))
    img_array = np.array(img)
    image_array= np.expand_dims(img_array, axis=0)
    result= model.predict(image_array)

    max_arg= np.argmax(result, axis=1)
    if max_arg[0]==0:
        st.image(image, caption='Prediction: CAT')
    else:
        st.image(image, caption='Prediction: DOG')

