#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import librosa
import pywt
import joblib
import paho.mqtt.client as mqtt
from sklearn.preprocessing import StandardScaler
import time

time.sleep(0.1)
# Constants for feature extraction
N_MFCC = 13
N_FFT = 512
HOP_LENGTH = 256

# Terbaik untuk scream 512 dan 256
# Terbaik untuk Conversation 1024 dan 256

# Load the trained SVM model
model_filename = 'D:\Data\Skripsi\ide 4 iot\PERCOBAAN SKRIPSI\PEMBUATAN PYTHON\python\suara_model_svm.pkl'
svm_model = joblib.load(model_filename)

# Function for denoising using wavelet
def denoise_wavelet(audio, threshold=1e-6, preserve_threshold=0.05):
    coeffs = pywt.wavedec(audio, 'db1', level=6)
    preserved_coeffs = [coeffs[0]]
    
    for i in range(1, len(coeffs)):
        thresholded_coeff = pywt.threshold(coeffs[i], threshold, mode='soft')
        if np.sum(np.abs(thresholded_coeff)) > preserve_threshold:
            preserved_coeffs.append(thresholded_coeff)
        else:
            preserved_coeffs.append(np.zeros_like(thresholded_coeff))

    audio_denoised = pywt.waverec(preserved_coeffs, 'db1')
    return audio_denoised

# Function to preprocess audio and extract features
def preprocess_audio(audio_data, sr=16000, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH):
    # Normalize sensor data to match the range of audio data
    audio = 2 * (audio_data - np.min(audio_data)) / (np.max(audio_data) - np.min(audio_data)) - 1

    # Denoising using wavelet
    audio_denoised = denoise_wavelet(audio, preserve_threshold=0.05)

    # Extracting features
    mfccs = librosa.feature.mfcc(y=audio_denoised, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    rmse = librosa.feature.rms(y=audio_denoised, frame_length=n_fft, hop_length=hop_length)
    spectral_centroid = librosa.feature.spectral_centroid(y=audio_denoised, sr=sr, n_fft=n_fft, hop_length=hop_length)

    # Calculate mean values
    mean_mfcc = np.mean(mfccs, axis=1)
    mean_rmse = np.mean(rmse)
    mean_centroid = np.mean(spectral_centroid)

    return np.concatenate([mean_mfcc, [mean_rmse], [mean_centroid]]).reshape(1, -1)

# Callback function for when the client receives a CONNACK response from the server.
def on_connect(client, userdata, flags, rc):
    print(f"Connected with result code {rc}")
    # Subscribe to the topic "topic/2/value"
    client.subscribe("topic/2/value")

# Callback function for when the client receives a message from the server.
def on_message(client, userdata, msg):
    print(f"Received message: {msg.payload}")
    # Convert the received array string to integer array
    received_array = np.array(msg.payload.decode("utf-8").split(','), dtype=int)
    # Extract the sensor code (last element)
    sensor_code = received_array[-1]
    # Remove the sensor code from the array used for prediction
    audio_data = received_array[:-1]
    # Preprocess the received audio data
    sound_features = preprocess_audio(audio_data)
    # Make prediction using the SVM model
    prediction = svm_model.predict(sound_features)
    # Display the classification result
    print('Sound Classification:', "Scream" if prediction[0] == 1 else "Conversation")
    # Combine prediction and sensor code into one message
    combined_message = f"{prediction[0]},{sensor_code}"
    # Publish the processed data to the topic "topic/1/value"
    client.publish("topic/1/value", combined_message)

# Set MQTT broker address
broker_address = "192.168.43.243"  # Ganti dengan alamat IP broker MQTT EMQX

# Create a MQTT client instance
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
# Set the callback functions
client.on_connect = on_connect
client.on_message = on_message

# Connect to the MQTT broker
client.connect(broker_address, 1883, 60)

# Start the MQTT loop
client.loop_forever()

