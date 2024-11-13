# NSFW Video Classification App

This project is a **Streamlit application** that allows users to upload a video and classify it as **NSFW** or **Safe** using various deep learning models. The app leverages models such as **I3D NSFW**, **3D CNN**, and **I3D** for video classification, providing predictions, confidence scores, and performance metrics.

## Table of Contents
- [Problem Statement](#problem-statement)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Models](#models)
- [Project Structure](#project-structure)
- [Contributors](#contributors)
- [License](#license)

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
