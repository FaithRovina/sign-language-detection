import pathlib
import platform

# Fix for Windows path issue
if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath

import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import tempfile
import os

# Set page config
st.set_page_config(
    page_title="Sign Language Detection",
    page_icon="👋",
    layout="wide"
)

# Title and description
st.title("Sign Language Detection")
st.write("Detect sign language in real-time or from uploaded videos")

# Sidebar for model options
st.sidebar.title("Options")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.01)
iou_threshold = st.sidebar.slider("IOU Threshold", 0.0, 1.0, 0.45, 0.01)

# Load YOLOv5 model from PyTorch Hub
@st.cache_resource
def load_model():
    # Clear cache and force reload
    torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True, trust_repo=True)
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()
model.conf = confidence_threshold
def process_frame(frame):
    # Convert frame to RGB (YOLOv5 expects RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Run inference
    results = model(frame_rgb)
    
    # Render results on frame
    rendered_frame = results.render()[0]
    
    # Convert back to BGR for display
    return cv2.cvtColor(rendered_frame, cv2.COLOR_RGB2BGR)

# Main app
def main():
    st.sidebar.title("Input Source")
    app_mode = st.sidebar.radio("Choose input source:", ["Webcam", "Upload Video"])
    
    if app_mode == "Webcam":
        st.header("Webcam Live Feed")
        run_webcam()
    else:
        st.header("Upload Video")
        run_video_upload()

def run_webcam():
    run = st.checkbox('Start Webcam')
    FRAME_WINDOW = st.image([])
    
    cap = cv2.VideoCapture(0)
    
    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture video from webcam.")
            break
            
        # Process frame
        processed_frame = process_frame(frame)
        
        # Display the processed frame
        FRAME_WINDOW.image(processed_frame, channels="BGR")
    
    cap.release()

def run_video_upload():
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])
    
    if uploaded_file is not None:
        # Save uploaded file to a temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        
        # Open video file
        cap = cv2.VideoCapture(tfile.name)
        
        # Create placeholders
        frame_placeholder = st.empty()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process frame
            processed_frame = process_frame(frame)
            
            # Display the processed frame
            frame_placeholder.image(processed_frame, channels="BGR")
        
        cap.release()
        os.unlink(tfile.name)

if __name__ == "__main__":
    main()
