import pathlib
import platform
import json
import os
from google.oauth2 import service_account
from google.cloud import speech
import sounddevice as sd
import queue
import threading
import numpy as np
import streamlit as st

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
import time
import torch

# Set page config with larger layout
st.set_page_config(
    page_title="Accessibility Assistant",
    page_icon="🤝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to adjust the main content width
st.markdown("""
    <style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        padding: 10px;
        margin: 5px 0;
    }
    .stVideo {
        max-width: 100% !important;
    }
    """, unsafe_allow_html=True)

# Global model options will be set in their respective pages
confidence_threshold = 0.25
iou_threshold = 0.45

# Initialize Google Cloud Speech client
def get_speech_client():
    try:
        credentials_path = r"C:\Users\faith\Downloads\speechtotext.json"
        credentials = service_account.Credentials.from_service_account_file(credentials_path)
        client = speech.SpeechClient(credentials=credentials)
        return client
    except Exception as e:
        st.error(f"Error initializing speech client: {str(e)}")
        return None

# Audio recording parameters
RATE = 16000
CHUNK = int(RATE / 10)  # 100ms

class AudioRecorder:
    def __init__(self):
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.recording = []
        self.stream = None
        self.thread = None

    def callback(self, indata, frames, time, status):
        if self.is_recording:
            self.audio_queue.put(indata.copy())

    def start_recording(self):
        self.is_recording = True
        self.recording = []
        self.stream = sd.InputStream(
            samplerate=RATE,
            channels=1,
            dtype='int16',
            callback=self.callback
        )
        self.stream.start()
        # Start processing thread
        self.thread = threading.Thread(target=self.process_queue, daemon=True)
        self.thread.start()

    def stop_recording(self):
        self.is_recording = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        # Process any remaining data in the queue
        self.process_queue()
        return np.concatenate(self.recording, axis=0) if self.recording else np.array([])

    def process_queue(self):
        while self.is_recording or not self.audio_queue.empty():
            try:
                data = self.audio_queue.get(timeout=0.5)
                if data is not None:
                    self.recording.append(data)
            except queue.Empty:
                if not self.is_recording:
                    break

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
def process_frame(frame, confidence_threshold=0.40, iou_threshold=0.45):
    try:
        # Convert frame to RGB (YOLOv5 expects RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Update model confidence and IOU thresholds
        model.conf = confidence_threshold
        model.iou = iou_threshold
        
        # Run inference
        with torch.no_grad():  # Disable gradient calculation for inference
            results = model(frame_rgb, size=320)  # Fixed size for faster processing
        
        # Get detections
        detections = results.pandas().xyxy[0]
        
        # Filter detections by confidence
        detections = detections[detections['confidence'] >= confidence_threshold]
        
        # Render results on frame
        rendered_frame = results.render()[0]
        
        # Convert back to BGR for display
        return cv2.cvtColor(rendered_frame, cv2.COLOR_RGB2BGR)
    except Exception as e:
        st.error(f"Error in frame processing: {str(e)}")
        return frame  # Return original frame if processing fails

# Main app
def transcribe_audio(audio_data, client):
    try:
        audio = speech.RecognitionAudio(content=audio_data.tobytes())
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=RATE,
            language_code="en-US",
            enable_automatic_punctuation=True,
        )
        response = client.recognize(config=config, audio=audio)
        return ' '.join([result.alternatives[0].transcript for result in response.results])
    except Exception as e:
        st.error(f"Error in live audio transcription: {str(e)}")
        return ""

def main_menu():
    st.title("Accessibility Assistant")
    st.write("Choose a mode to get started:")
    
    # Clear any existing sidebar content
    st.sidebar.empty()
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🎤 Speech Recognition", use_container_width=True):
            st.session_state.page = "speech"
            st.rerun()
    
    with col2:
        if st.button("👋 Sign Language Detection", use_container_width=True):
            st.session_state.page = "sign_language"
            st.rerun()

def transcribe_audio_file(audio_file, client):
    try:
        # Read the audio file
        audio_content = audio_file.read()
        
        # Check file type and set encoding
        file_extension = audio_file.name.split('.')[-1].lower()
        
        # Map common audio formats to their respective encodings
        encoding_map = {
            'wav': speech.RecognitionConfig.AudioEncoding.LINEAR16,
            'flac': speech.RecognitionConfig.AudioEncoding.FLAC,
            'mp3': speech.RecognitionConfig.AudioEncoding.MP3,
            'ogg': speech.RecognitionConfig.AudioEncoding.OGG_OPUS,
        }
        
        # Default to LINEAR16 if format not recognized
        encoding = encoding_map.get(file_extension, speech.RecognitionConfig.AudioEncoding.ENCODING_UNSPECIFIED)
        
        # Create audio object
        audio = speech.RecognitionAudio(content=audio_content)
        
        # Create config
        config = speech.RecognitionConfig(
            encoding=encoding,
            sample_rate_hertz=RATE,
            language_code="en-US",
            enable_automatic_punctuation=True,
        )
        
        # Make the API request
        response = client.recognize(config=config, audio=audio)
        
        # Combine all results
        return ' '.join([result.alternatives[0].transcript for result in response.results])
    except Exception as e:
        st.error(f"Error in file transcription: {str(e)}")
        return ""

def speech_page():
    st.title("🎤 Speech Recognition")
    
    # Clear any existing sidebar content
    st.sidebar.empty()
    
    # Add back button
    if st.button("⬅️ Back to Main Menu"):
        st.session_state.page = "main"
        st.rerun()
    
    # Initialize speech client
    speech_client = get_speech_client()
    
    # Initialize recorder if not exists
    if 'recorder' not in st.session_state:
        st.session_state.recorder = AudioRecorder()
    
    # Initialize transcription if not exists
    if 'transcription' not in st.session_state:
        st.session_state.transcription = ""
    
    # Tab layout for different input methods
    tab1, tab2 = st.tabs(["🎤 Record Audio", "📁 Upload Audio File"])
    
    with tab1:
        st.write("Record your voice and see the transcription:")
        
        # Recording controls
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🎤 Start Recording", key="start_recording"):
                st.session_state.recording = True
                st.session_state.recorder.start_recording()
                st.session_state.recording_started = True
                st.toast('Recording started...', icon='🎤')
        
        with col2:
            if st.button("⏹️ Stop Recording", key="stop_recording"):
                if 'recording_started' in st.session_state and st.session_state.recording_started:
                    audio_data = st.session_state.recorder.stop_recording()
                    if speech_client is not None and len(audio_data) > 0:
                        with st.spinner('Transcribing...'):
                            st.session_state.transcription = transcribe_audio(audio_data, speech_client)
                    st.session_state.recording_started = False
                    st.toast('Recording stopped', icon='⏹️')
    
    with tab2:
        st.write("Upload an audio file for transcription (supports WAV, MP3, FLAC, OGG):")
        uploaded_file = st.file_uploader("Choose an audio file", 
                                       type=['wav', 'mp3', 'flac', 'ogg'],
                                       key="audio_uploader")
        
        if uploaded_file is not None:
            if st.button("Transcribe Audio File", key="transcribe_file"):
                if speech_client is not None:
                    with st.spinner('Transcribing file...'):
                        st.session_state.transcription = transcribe_audio_file(uploaded_file, speech_client)
    
    # Display transcription
    st.subheader("Transcription")
    st.text_area("Transcription", 
                value=st.session_state.transcription, 
                height=200,
                key="transcription_area")

def sign_language_page():
    st.title("👋 Sign Language Detection")
    
    # Add back button
    if st.button("⬅️ Back to Main Menu"):
        st.session_state.page = "main"
        st.rerun()
    
    # Model options in sidebar
    st.sidebar.title("Model Options")
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.01, key="conf_thresh_sign")
    iou_threshold = st.sidebar.slider("IOU Threshold", 0.0, 1.0, 0.45, 0.01, key="iou_thresh_sign")
    
    # Input source selection
    st.sidebar.title("Input Source")
    app_mode = st.sidebar.radio("Choose input source:", ["Webcam", "Upload Video"], key="input_source")
    
    if app_mode == "Webcam":
        st.header("Webcam Live Feed")
        run_webcam()
    else:
        st.header("Upload Video")
        run_video_upload()

def main():
    # Initialize session state for page navigation
    if 'page' not in st.session_state:
        st.session_state.page = "main"
    
    # Page routing
    if st.session_state.page == "main":
        main_menu()
    elif st.session_state.page == "speech":
        speech_page()
    elif st.session_state.page == "sign_language":
        sign_language_page()

def run_webcam():
    # Get current confidence and IOU thresholds from session state
    conf_threshold = st.session_state.get('conf_thresh_sign', 0.40)  # Increased default confidence
    iou_threshold = st.session_state.get('iou_thresh_sign', 0.45)
    
    # Add performance options
    st.sidebar.subheader("Performance Settings")
    frame_skip = st.sidebar.slider("Frame Skip", 1, 5, 2, 1,
                                 help="Process every nth frame to improve performance")
    
    # Add model settings
    model_size = st.sidebar.selectbox(
        "Model Size",
        ["nano", "small", "medium"],
        index=1,
        help="Larger models are more accurate but slower"
    )
    
    # Load appropriate model based on selection
    @st.cache_resource
    def load_model(size="small"):
        # Clear cache and force reload
        torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
        model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True, trust_repo=True)
        
        # Set model parameters based on size
        if size == "nano":
            model = model.autoshape()  # Smallest model
            model.conf = max(conf_threshold, 0.5)  # Higher confidence threshold for smaller model
        elif size == "medium":
            model.conf = conf_threshold
            model = model.fuse()  # Fuse layers for better performance
        else:  # small
            model.conf = conf_threshold
            
        return model
    
    model = load_model(model_size)
    
    run = st.checkbox('Start Webcam', key='webcam_checkbox')
    FRAME_WINDOW = st.image([])
    
    cap = cv2.VideoCapture(0)
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    frame_count = 0
    fps = st.empty()
    
    # Warm-up the model
    _ = model(torch.zeros(1, 3, 320, 320).to('cuda' if torch.cuda.is_available() else 'cpu'))
    
    try:
        while run:
            start_time = time.time()
            
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture video from webcam.")
                break
                
            frame_count += 1
            
            # Only process every nth frame
            if frame_count % frame_skip == 0:
                # Resize frame for faster processing
                small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                
                # Process frame with current thresholds
                processed_frame = process_frame(small_frame, model.conf, iou_threshold)
                
                # Resize back to original size for display
                processed_frame = cv2.resize(processed_frame, (frame.shape[1], frame.shape[0]))
                
                # Calculate and display FPS
                fps_text = f"FPS: {1.0 / (time.time() - start_time):.1f}"
                cv2.putText(processed_frame, fps_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Display the processed frame
                FRAME_WINDOW.image(processed_frame, channels="BGR")
            
            # Add a small delay to prevent high CPU usage
            time.sleep(0.01)
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

def run_video_upload():
    # Get current confidence and IOU thresholds from session state
    conf_threshold = st.session_state.get('conf_thresh_sign', 0.25)
    iou_threshold = st.session_state.get('iou_thresh_sign', 0.45)
    
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"], key="video_uploader")
    
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
                
            # Process frame with current thresholds
            processed_frame = process_frame(frame, conf_threshold, iou_threshold)
            
            # Display the processed frame
            frame_placeholder.image(processed_frame, channels="BGR")
        
        cap.release()
        os.unlink(tfile.name)

if __name__ == "__main__":
    main()
