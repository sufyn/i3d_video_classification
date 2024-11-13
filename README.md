# NSFW Video Classification App

This project is a **Streamlit application** that allows users to upload a video and classify it as **NSFW** or **Safe** using various deep learning models. The app leverages models such as **I3D NSFW**, **3D CNN**, and **I3D** for video classification, providing predictions, confidence scores, and performance metrics.

## Table of Contents
- [Problem Statement](#problem-statement)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Models](#models)
- [Project Structure](#project-structure)

## Problem Statement
The project addresses the challenge of automating **content moderation** for videos by classifying them as `NSFW` or `Safe`. This is useful for various applications in social media, content-sharing platforms, and any system where user-uploaded video content needs moderation.

## Features
- **Video Classification**: Classifies videos as NSFW or Safe.
- **Model Selection**: Choose from multiple pre-trained models (I3D NSFW, 3D CNN, and I3D).
- **Confidence Score**: Shows the model's confidence in its classification.
- **Performance Metrics**: Displays the time taken for prediction and CPU usage.
- **Frame Extraction Preview**: Visualizes video frames for additional insight into model predictions.

## Installation
1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/nsfw-video-classification-app.git
   cd nsfw-video-classification-app
   ```
Download model files: The application will automatically download the necessary model files upon first run. You may also manually download and place them in the project directory.

## Usage
Run the Streamlit app:

``` bash

streamlit run app.py
```
Upload a video file: In the Streamlit interface, upload a video in .mp4, .avi, or .mov format.

Select a model: Choose a model from the dropdown menu.

Predict: Click on the "Predict" button to classify the video. The app will display the classification result, confidence score, and performance metrics.

## Models
The app provides three model options for video classification:

I3D NSFW: Pre-trained I3D model for detecting NSFW content.

3D CNN: A 3D Convolutional Neural Network model.

I3D: General-purpose I3D model for broader video classification tasks.

Model Files

Model files are automatically downloaded from the URLs specified in the code. They include:

nsfw_i3d_model.h5

3dcnn_model.h5

i3d_model.h5

## Project Structure
``` bash
nsfw-video-classification-app/
│
├── app.py                       # Main Streamlit app code
├── requirements.txt             # List of dependencies
├── README.md                    # Project documentation
├── models/                      # Directory for downloaded model files
└── utils.py                     # Helper functions for frame extraction and model loading
```
