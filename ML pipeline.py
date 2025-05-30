# import main libraires and tools

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from config import *
from PIL import Image
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error
from tensorflow.keras import layers, models, losses
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# loading need saved models : autoencoder for detecting anomaly and the cnn for classifying

autoencoder = load_model('d:/AI Projects/Signals detecting and classification system/models/autoencoder_model.keras')
classifier = load_model('d:/AI Projects/Signals detecting and classification system/models/classifier_model.h5')

# function for predictiong the anomaly signals above the threshold level
def detect_anomaly_fft(image, THRESHOLD):
    input_image = np.expand_dims(image, axis=0)
    reconstructed = autoencoder.predict(input_image, verbose=0)[0]
    error = mean_squared_error(image.flatten(), reconstructed.flatten())
    print("Reconstruction Error:", error)
    return error, error > THRESHOLD




# function for classifying the images if not anomaly
def classify_fft_image(image):
    input_image = np.expand_dims(image, axis=0)
    pred = classifier.predict(input_image, verbose=0)[0]
    class_idx = np.argmax(pred)
    confidence = pred[class_idx]
    return class_idx, confidence



'''the complete pipeline for the ml model at first loading images , passing to autoencoder at first 
    then to cnn for clasify'''

def process_fft_pipeline(image_path, THRESHOLD, classifier_model=None, class_names=None):


    #  Load, Resize and Normalize
    image = Image.open(image_path).convert('L').resize((64, 64))
    image = np.array(image).astype("float32") / 255.0
    image = np.expand_dims(image, axis=-1)  # (64, 64, 1)
    input_image = np.expand_dims(image, axis=0)  # (1, 64, 64, 1)

    #  Anomaly Detection
    reconstructed = autoencoder.predict(input_image, verbose=0)[0]
    error = mean_squared_error(image.flatten(), reconstructed.flatten())
    

    print(f"Reconstruction Error: {error:.5f}")

    if error > THRESHOLD:
        print(" Anomalous Signal Detected!")
        print("_________________________________________\n")
        return "anomaly", None

    else:
        print("Normal Signal")

        if classifier_model is not None:
            preds = classifier_model.predict(input_image, verbose=0)[0]
            class_idx = np.argmax(preds)
            class_name = class_names[class_idx] if class_names else f"class_{class_idx}"
            confidence = preds[class_idx] * 100

            print(f"Signal Type Detected: {class_name} , acc={confidence:.2f}%")
            print("\n_________________________________________\n")
            return "normal", class_name
        else:
            print(" No classifier model loaded.")
            print("\n_________________________________________\n")
            return "normal", None


# to know the classify number we make a list for every number with the facing signal type

classifier_model = classifier


# '''we make a dictionary for testing signals , they were 10 images , 
# we loaded them for testing them with the desired information'''

for root, folder, images in os.walk('Test signals'):
    for image in images:
        if image.endswith(('.png', '.jpg')):
            img_path = os.path.join(root, image)
            status, signal_type =process_fft_pipeline(
            img_path,
            Threshold,
            classifier_model=classifier_model,
            class_names=Signals
            )




