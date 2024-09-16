import streamlit as st
import tensorflow as tf
import requests
import os
import numpy as np
import cv2
import time
import psutil
import matplotlib.pyplot as plt
# Define the URL to the file in GitHub Releases
model_url = 'https://github.com/sufyn/i3d_video_classification/releases/download/nsfw/nsfw_i3d_model.h5'

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

# Define frame extraction function
def extract_frames(video_path, num_frames=16):
    """
    Extract frames from a video file.

    Args:
    - video_path (str): Path to the video file.
    - num_frames (int): Number of frames to extract (default is 16).

    Returns:
    - frames (np.array): Array of extracted frames.
    """
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

# Define prediction function
def predict_video(video_path, model):
    """
    Predicts the class of a video using the given model.

    Args:
    - video_path (str): Path to the video file.
    - model (tf.keras.Model): The trained model.

    Returns:
    - predicted_class (str): Predicted class label.
    - confidence_score (float): Confidence score of the prediction.
    - prediction_time (float): Time taken for prediction.
    - cpu_usage (float): CPU usage during prediction.
    """
    start_time = time.time()

    # Extract frames from the video
    frames = extract_frames(video_path)

    # Make a prediction
    prediction = model.predict(np.expand_dims(frames, axis=0))
    predicted_class_index = np.argmax(prediction)
    confidence_score = np.max(prediction)

    # Class labels
    class_labels = ['nsfw', 'safe']  # Adjust based on your dataset
    predicted_class = class_labels[predicted_class_index]

    end_time = time.time()
    prediction_time = end_time - start_time

    # Get CPU usage
    cpu_usage = psutil.cpu_percent()

    return predicted_class, confidence_score, prediction_time, cpu_usage

# Streamlit App UI
st.title("NSFW Video Classification App")
st.write("Upload a video and classify it as NSFW or Safe.")

# Upload video file
uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # Save uploaded video to a temporary file
    with open(os.path.join("temp_video.mp4"), "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.video(uploaded_file)

    # Perform prediction when button is clicked
    if st.button("Predict"):
        # Run the prediction
        predicted_class, confidence_score, prediction_time, cpu_usage = predict_video("temp_video.mp4", model_i3d)

        # Display prediction results
        st.write(f"**Predicted Class:** {predicted_class}")
        st.write(f"**Confidence Score:** {confidence_score:.4f}")
        st.write(f"**Time Taken for Prediction:** {prediction_time:.4f} seconds")
        st.write(f"**CPU Usage:** {cpu_usage:.2f}%")

        # Extract and display video frames
        st.write("Video Frames:")
        frames = extract_frames("temp_video.mp4")
        fig, ax = plt.subplots(4, 4, figsize=(10, 5))
        for i in range(len(frames)):
            ax[i // 4, i % 4].imshow(frames[i])
            ax[i // 4, i % 4].axis('off')
        st.pyplot(fig)

# Footer
st.write("Model: I3D for NSFW/Safe Classification")
