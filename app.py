import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
from pathlib import Path

# Add yolov5 to path
import sys
sys.path.append('yolov5')  # path to the cloned repo

# Load the YOLOv5 model
from models.experimental import attempt_load

# Set device (GPU if available, otherwise CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = attempt_load('best.pt', device=device)

# Get class names
with open('yolov5/data/sign_data.yaml', 'r') as f:
    classes = yaml.safe_load(f)['names']

st.title("Sign Language Detection")

# Upload image
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Convert the file to an opencv image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    
    # Convert the image to RGB (OpenCV uses BGR by default)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    
    # Create a PIL image from the opencv image
    image = Image.fromarray(opencv_image)
    
    # Display the image
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Run inference
    if st.button('Detect Signs'):
        # Convert PIL image to numpy array
        img_array = np.array(image)
        
        # Run inference
        results = model(img_array)
        
        # Display results
        st.image(results.render(), caption='Detected Signs', use_column_width=True)
        
        # Display detection details
        st.write("Detection Details:")
        for detection in results.pandas().xyxy[0].itertuples():
            st.write(f"Class: {detection.name}, Confidence: {detection.confidence:.2f}")

# Add a video upload option
uploaded_video = st.file_uploader("Choose a video", type=["mp4", "avi"])

if uploaded_video is not None:
    video_bytes = uploaded_video.read()
    
    # Create a temporary file for the video
    with open("temp_video.mp4", "wb") as f:
        f.write(video_bytes)
    
    # Run inference on video
    if st.button('Process Video'):
        cap = cv2.VideoCapture("temp_video.mp4")
        
        # Create output video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert frame to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Run inference
            results = model(frame_rgb)
            
            # Get the rendered frame
            rendered_frame = results.render()[0]
            
            # Write the frame to output video
            out.write(cv2.cvtColor(rendered_frame, cv2.COLOR_RGB2BGR))
        
        cap.release()
        out.release()
        
        # Display the processed video
        st.video("output.mp4")
