import streamlit as st
import tensorflow as tf
import requests
import os
import numpy as np
import cv2
import time
import psutil

# Define the URL to the file in GitHub Releases
model_url = 'https://github.com/yourusername/yourrepo/releases/download/v1.0/nsfw_i3d_model.h5'

# Download the file
model_path = 'nsfw_i3d_model.h5'

def download_file(url, local_path):
    r = requests.get(url, allow_redirects=True)
    with open(local_path, 'wb') as f:
        f.write(r.content)

# Check if the model file is already downloaded
if not os.path.isfile(model_path):
    st.write('Downloading model file...')
    download_file(model_url, model_path)
    st.write('Download complete!')

# Load the model
model_i3d = tf.keras.models.load_model(model_path)

def extract_frames(video_path, num_frames=16):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(total_frames // num_frames, 1)

    for i in range(num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * frame_interval)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (112, 112))
        frames.append(frame)

    cap.release()

    while len(frames) < num_frames:
        frames.append(np.zeros((112, 112, 3), np.uint8))

    return np.array(frames)

def predict_video(video_path, model):
    start_time = time.time()
    frames = extract_frames(video_path)
    prediction = model.predict(np.expand_dims(frames, axis=0))
    predicted_class_index = np.argmax(prediction)
    class_labels = ['nsfw', 'safe']
    predicted_class = class_labels[predicted_class_index]
    end_time = time.time()
    prediction_time = end_time - start_time
    cpu_usage = psutil.cpu_percent()
    return predicted_class, prediction_time, cpu_usage

# Streamlit UI
st.title('Video Classification')

uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    with open("temp_video.mp4", "wb") as f:
        f.write(uploaded_file.getvalue())
    
    st.video("temp_video.mp4")
    
    predicted_class, prediction_time, cpu_usage = predict_video("temp_video.mp4", model_i3d)
    
    st.write(f"Predicted class: {predicted_class}")
    st.write(f"Prediction time: {prediction_time:.4f} seconds")
    st.write(f"CPU usage: {cpu_usage:.2f}%")
