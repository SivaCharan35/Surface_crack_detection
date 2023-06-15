import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras as keras
import tensorflow.keras.preprocessing.image as image
import time
from PIL import Image

st.set_page_config(
    page_title="Concrete Surface Crack Detector"
)

st.markdown(
    """
    <style>
    [data-testid="stFileUploadDropzone"] {
        margin-top: -30px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.write("## Concrete Surface Crack Detector")

st.write("## Upload Your Image")
img = st.file_uploader("", type=["jpg"])

saved_model = keras.models.load_model("cnnfinal.h5")

if st.button("Submit"):
    st.write("")
    my_bar = st.progress(0)
    for percent_complete in range(100):
        time.sleep(0.1)
        my_bar.progress(percent_complete + 1)
    
    if img is not None:
        img = Image.open(img)

        # Resize the image while maintaining aspect ratio
        img = img.resize((224, 224))

        img_array = image.img_to_array(img)
        img_array /= 255.0  # Rescale the image by dividing by 255
        img_batch = np.expand_dims(img_array, axis=0)

        # Make predictions using the loaded model
        predictions = saved_model.predict(img_batch)

        # Print the predicted class label
        if predictions[0] < 0.50:
            st.write("## The Image Doesn't Contain Crack!")
        else:
            st.write("## The Image Contains Crack!")
        
        # Display the uploaded image
        st.image(img, caption='Uploaded Image', use_column_width=True)
    else:
        st.write("## No image uploaded. Please upload an image.")
