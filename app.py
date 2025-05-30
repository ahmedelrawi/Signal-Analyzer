# import main libraires and tools
import streamlit as st
import numpy as np
from PIL import Image
from config import *
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error


# Load models

autoencoder = load_model('d:/AI Projects/Signals detecting and classification system/models/autoencoder_model.keras')
classifier = load_model('d:/AI Projects/Signals detecting and classification system/models/classifier_model.h5')


# two function for detecting and classifying the signals

def detect_anomaly(img_array):
    img_norm = np.expand_dims(img_array, axis=0)  # (1, 64, 64, 1)
    reconstructed = autoencoder.predict(img_norm, verbose=0)[0]
    error = mean_squared_error(img_array.flatten(), reconstructed.flatten())
    return error, error > Threshold



def classify_signal(img_array):
    img_norm = np.expand_dims(img_array, axis=0)
    preds = classifier.predict(img_norm, verbose=0)[0]
    class_id = np.argmax(preds)
    return Signals[class_id], preds[class_id] * 100



# Streamlit UI

st.title("Signal Analyzer and Classifier")
uploaded_file = st.file_uploader("Upload FFT Signal Image", type=['png', 'jpg'])



if uploaded_file:
    img = Image.open(uploaded_file).convert('L').resize((64, 64))
    img = np.array(img).astype("float32")/255.0
    st.image(img, caption="Uploaded Signal", use_column_width=True)

    error, is_anomaly = detect_anomaly(np.expand_dims(img, axis=-1))
    st.write(f"Reconstruction Error: `{error:.5f}`")

    if is_anomaly:
        st.error("Anomalous Signal Detected!")
    else:
        signal_type, confidence = classify_signal(np.expand_dims(img, axis=-1))
        st.success(f"Normal Signal Detected - Type: **{signal_type}** ({confidence:.2f}%)")
