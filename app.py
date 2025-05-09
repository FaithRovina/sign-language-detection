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

# --- GLOBAL CSS FOR CONSISTENT FONTS AND LAYOUT ---
st.markdown("""
    <style>
    html, body, [class^='css'] {
        font-family: 'Segoe UI', 'Roboto', 'Arial', sans-serif !important;
        font-size: 20px !important;
        color: #222 !important;
    }
    .stTextInput input, .stTextArea textarea, .stButton button, .stTabs [data-baseweb="tab"], .stSelectbox label, .stSelectbox div[data-baseweb="select"] {
        font-size: 20px !important;
        font-family: 'Segoe UI', 'Roboto', 'Arial', sans-serif !important;
    }
    .block-container {
        max-width: 900px;
        margin-left: auto;
        margin-right: auto;
        padding-top: 2.5em;
        padding-bottom: 2em;
    }
    .instructions-block {
        background: #f1f3f6;
        border-radius: 12px;
        padding: 1.5em;
        margin-bottom: 1.5em;
        font-size: 22px !important;
        color: #222;
    }
    .visual-aid-category {
        font-size: 1.2em;
        font-weight: bold;
        margin-bottom: 0.5em;
        margin-top: 1em;
        text-align: center;
    }
    @media (max-width: 600px) {
        .block-container {
            max-width: 98vw !important;
            padding: 0.5em !important;
        }
        .stSelectbox label, .stSelectbox div[data-baseweb="select"] {
            font-size: 1.1em !important;
        }
        img {
            max-width: 98vw !important;
            height: auto !important;
        }
    }
    </style>
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

def chat_interface():
    st.title("💬 Two-Person Chat")
    
    # Initialize chat history if not exists
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'current_speaker' not in st.session_state:
        st.session_state.current_speaker = "Person 1"
    if 'show_help' not in st.session_state:
        st.session_state.show_help = False
    
    # --- Custom Help/Instructions Toggle ---
    if 'show_instructions' not in st.session_state:
        st.session_state.show_instructions = False

    help_col, _ = st.columns([1, 8])
    with help_col:
        if st.button('❓', key='chat_help_toggle', help='Show/Hide Instructions', use_container_width=False):
            st.session_state.show_instructions = not st.session_state.show_instructions

    if st.session_state.show_instructions:
        st.markdown(
            '''<div class="instructions-block">
            <b>How to Use:</b><br>
            - <b>Two-Person Chat:</b> Person 1 and Person 2 take turns using this device.<br>
            - <b>Current Speaker:</b> The current speaker is highlighted.<br>
            - <b>Type Message:</b> Type your message and press Enter or click Send.<br>
            - <b>Switch Speaker:</b> Click "Switch Speaker" to change turns.<br>
            - <b>Clear Chat:</b> Use the "Clear Chat" button to start a new conversation.<br>
            Click the ❓ again to hide these instructions.
            </div>''',
            unsafe_allow_html=True
        )
    
    # Display chat messages with speaker indicators
    st.markdown("---")
    chat_container = st.container()
    
    # Define speaker colors
    speaker_colors = {
        "Person 1": {
            "icon": "#4CAF50",  # Green
            "text": "#2E7D32"  # Darker green
        },
        "Person 2": {
            "icon": "#2196F3",  # Blue
            "text": "#1565C0"  # Darker blue
        }
    }
    
    with chat_container:
        for message in st.session_state.chat_history:
            if message["role"] == "system":
                st.markdown(f"*{message['content']}*")
            else:
                speaker = message["speaker"]
                color = speaker_colors.get(speaker, {"icon": "#9E9E9E", "text": "#424242"})
                
                with st.chat_message(message["role"]):
                    st.markdown(
                        f"<span style='color: {color['text']}; font-weight: bold;'>{speaker}:</span> {message['content']}",
                        unsafe_allow_html=True
                    )
    
    # Chat controls
    col1, col2, col3 = st.columns([3, 2, 2])
    
    with col1:
        # Chat input
        prompt = st.text_input("Type your message here...", 
                             key=f"chat_input_{st.session_state.current_speaker}",
                             label_visibility="collapsed")
    
    with col2:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)  # Spacer
        if st.button("✉️ Send"):
            if prompt.strip():
                # Add message to history
                st.session_state.chat_history.append({
                    "role": "user",
                    "speaker": st.session_state.current_speaker,
                    "content": prompt
                })
                st.rerun()
    
    with col3:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)  # Spacer
        if st.button("🔄 Switch Speaker"):
            st.session_state.current_speaker = "Person 2" if st.session_state.current_speaker == "Person 1" else "Person 1"
            st.rerun()
    
    # Current speaker indicator with theme-aware styling and dynamic colors
    speaker_colors = {
        "Person 1": {
            "icon": "#4CAF50",  # Green
            "text": "#2E7D32"   # Darker green
        },
        "Person 2": {
            "icon": "#2196F3",  # Blue
            "text": "#1565C0"   # Darker blue
        }
    }
    
    current_speaker = st.session_state.current_speaker
    speaker_color = speaker_colors.get(current_speaker, {"icon": "#9E9E9E", "text": "#424242"})
    
    # Create a container for the speaker indicator with inline styles
    st.markdown(
        f"""
        <div style="
            background-color: {'#f0f2f6' if not st.get_option('theme.base') == 'dark' else '#1e1e1e'};
            color: {'#31333F' if not st.get_option('theme.base') == 'dark' else '#f0f2f6'};
            padding: 12px 16px;
            border-radius: 8px;
            margin: 12px 0;
            border: 1px solid {'#e6e9ef' if not st.get_option('theme.base') == 'dark' else '#2b2b2b'};
            display: flex;
            align-items: center;
            gap: 10px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        ">
            <span style="font-size: 1.2em; color: {speaker_color['icon']};">🎤</span>
            <span>Now speaking: <span style="font-weight: bold; color: {speaker_color['text']};">{current_speaker}</span></span>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Additional controls
    col_clear, col_back = st.columns(2)
    
    with col_clear:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.session_state.current_speaker = "Person 1"
            st.rerun()
    
    with col_back:
        if st.button("⬅️ Back to Main Menu"):
            st.session_state.page = "main"
            st.rerun()

def main_menu():
    st.title("Accessibility Assistant")
    st.markdown("<div style='font-size:1.1em; margin-bottom:1.5em; color:#555; text-align:center;'>Making communication simple, accessible, and inclusive one sign, one gesture at a time.</div>", unsafe_allow_html=True)
    st.write("Choose a mode to get started:")
    
    st.sidebar.empty()
    
    # Four columns for the buttons (equal width)
    cols = st.columns(4)
    button_labels = [
        ("🎤 Speech Recognition", "speech"),
        ("👋 Sign Language", "sign_language"),
        ("💬 Text Chat", "chat"),
        ("🖼️ Visual Aid", "visual_aid")
    ]
    for i, (label, page) in enumerate(button_labels):
        with cols[i]:
            if st.button(label, use_container_width=True, key=f"mainbtn_{page}"):
                st.session_state.page = page
                st.rerun()
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    with st.expander("ℹ️ About the Features"):
        st.markdown("""
        - **🎤 Speech Recognition**: Convert your speech to text in real-time
        - **👋 Sign Language**: Detect sign language using your webcam or upload a video
        - **💬 Two-Person Chat**: Enable text-based communication between two people using the same device
        - **🖼️ Visual Aid**: Select pictograms to visually communicate messages
        """)


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

    # Custom CSS for even bigger font size (body and instructions)
    st.markdown(
        '''<style>
        html, body, [class^="css"]  {
            font-size: 24px !important;
        }
        .stTextInput input, .stTextArea textarea, .stButton button, .stTabs [data-baseweb="tab"] {
            font-size: 24px !important;
        }
        .instructions-block {
            background: #f1f3f6;
            border-radius: 12px;
            padding: 1.5em;
            margin-bottom: 1.5em;
            font-size: 28px !important;
            color: #222;
        }
        .help-toggle {
            cursor: pointer;
            font-size: 2.2em;
            margin-bottom: 0.5em;
            color: #1976d2;
            background: none;
            border: none;
            outline: none;
        }
        </style>''',
        unsafe_allow_html=True
    )

    # --- Custom Help/Instructions Toggle ---
    if 'show_instructions' not in st.session_state:
        st.session_state.show_instructions = False

    help_col, _ = st.columns([1, 8])
    with help_col:
        if st.button('❓', key='help_toggle', help='Show/Hide Instructions', use_container_width=False):
            st.session_state.show_instructions = not st.session_state.show_instructions

    if st.session_state.show_instructions:
        st.markdown(
            '''<div class="instructions-block">
            <b>How to Use:</b><br>
            - <b>Record Audio:</b> Click "Start Recording" to begin, then "Stop Recording" to finish and transcribe.<br>
            - <b>Upload Audio File:</b> Upload a supported audio file and click "Transcribe Audio File".<br>
            Your transcription will appear below.<br>
            Click the ❓ again to hide these instructions.
            </div>''',
            unsafe_allow_html=True
        )

    # Hide sidebar for this page
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
                height=200,)



def sign_language_page():
    st.title("👋 Sign Language Detection")
    
    # Add back button
    if st.button("⬅️ Back to Main Menu", key="back_btn", use_container_width=True):
        st.session_state.page = "main"
        st.rerun()

    # --- Main page controls ---
    st.markdown("<h3 style='font-size:1.5em;'>Model Options</h3>", unsafe_allow_html=True)
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.01, key="conf_thresh_sign")
    iou_threshold = st.slider("IOU Threshold", 0.0, 1.0, 0.45, 0.01, key="iou_thresh_sign")
    frame_skip = st.slider("Frame Skip", 1, 5, 2, 1, help="Process every nth frame to improve performance", key="frame_skip_sign")
    
    st.markdown("<h3 style='font-size:1.5em;'>Input Source</h3>", unsafe_allow_html=True)
    app_mode = st.radio("Choose input source:", ["Webcam", "Upload Video"], key="input_source", horizontal=True)
    
    if app_mode == "Webcam":
        st.header("Webcam Live Feed")
        run_webcam(confidence_threshold, iou_threshold, frame_skip)
    else:
        st.header("Upload Video")
        run_video_upload(confidence_threshold, iou_threshold, frame_skip)


# Hide Streamlit sidebar and hamburger menu globally
st.markdown("""
    <style>
    [data-testid="stSidebar"], [data-testid="stSidebarNav"] {display: none !important;}
    [data-testid="collapsedControl"] {display: none !important;}
    .visual-aid-img {
        border: 3px solid transparent;
        border-radius: 12px;
        transition: border-color 0.2s;
        cursor: pointer;
        margin-bottom: 8px;
    }
    .visual-aid-img.selected {
        border-color: #2196F3;
        box-shadow: 0 0 8px #2196F3;
    }
    .visual-aid-category {
        font-size: 1.2em;
        font-weight: bold;
        margin-bottom: 0.5em;
    }
    </style>
""", unsafe_allow_html=True)


def visual_aid_page():
    import glob
    st.title("🖼️ Visual Aid - Pictogram Board")
    st.markdown('<div class="visual-aid-category">Select a pictogram category and view the image below.</div>', unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    # Back button
    if st.button("⬅️ Back to Main Menu", key="visualaid_back", use_container_width=True):
        st.session_state.page = "main"
        st.rerun()

    pictograms_dir = os.path.join(os.getcwd(), "Pictograms")
    categories = ["CommonRequests", "Feelings", "Numbers"]
    category_labels = ["Common Requests", "Feelings", "Numbers"]
    cat_map = dict(zip(categories, category_labels))

    if 'visual_aid_category' not in st.session_state:
        st.session_state.visual_aid_category = categories[0]
    if 'visual_aid_selected' not in st.session_state:
        st.session_state.visual_aid_selected = None

    # --- Category selection as buttons ---
    st.markdown('<div class="visual-aid-category">Select a category:</div>', unsafe_allow_html=True)
    cat_cols = st.columns(len(categories))
    for i, (cat, label) in enumerate(zip(categories, category_labels)):
        if cat_cols[i].button(label, key=f"catbtn_{cat}", use_container_width=True):
            st.session_state.visual_aid_category = cat
            st.session_state.visual_aid_selected = None

    selected_category = st.session_state.visual_aid_category

    # List images in the selected category
    category_path = os.path.join(pictograms_dir, selected_category)
    image_files = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.svg"):
        image_files.extend(glob.glob(os.path.join(category_path, ext)))
    image_files = sorted(image_files)

    if not image_files:
        st.warning("No pictograms found in this category.")
        return

    # Spread out the content using columns
    col_left, col_center, col_right = st.columns([1,2,1])
    with col_center:
        st.markdown('<div class="visual-aid-category">Select a pictogram:</div>', unsafe_allow_html=True)
        names = [os.path.splitext(os.path.basename(f))[0] for f in image_files]
        name_to_path = {os.path.splitext(os.path.basename(f))[0]: f for f in image_files}
        selected_name = st.selectbox("Pictogram Name", names, key=f"picdropdown_{selected_category}")
        st.session_state.visual_aid_selected = name_to_path[selected_name]

    st.markdown("<br>", unsafe_allow_html=True)
    # Show the selected image in the center, with more vertical spacing
    if st.session_state.visual_aid_selected:
        st.markdown("<div style='display:flex; justify-content:center; align-items:center; margin-top:2em; margin-bottom:2em;'>", unsafe_allow_html=True)
        st.image(st.session_state.visual_aid_selected, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.success("This pictogram is selected for communication.")

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
    elif st.session_state.page == "chat":
        chat_interface()
    elif st.session_state.page == "visual_aid":
        visual_aid_page()

def run_webcam(conf_threshold, iou_threshold, frame_skip):
    # Load the model
    @st.cache_resource
    def load_model():
        # Clear cache and force reload
        torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
        model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True, trust_repo=True)
        model.conf = conf_threshold
        return model
    
    model = load_model()
    
    run = st.checkbox('Start Webcam', key='webcam_checkbox')
    FRAME_WINDOW = st.image([])
    
    cap = cv2.VideoCapture(0)
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    frame_count = 0
    fps = st.empty()
    
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
                # Process frame with current thresholds
                processed_frame = process_frame(frame, conf_threshold, iou_threshold)
                
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


def run_video_upload(conf_threshold, iou_threshold, frame_skip):
    # Get current confidence and IOU thresholds from session state
    conf_threshold = st.session_state.get('conf_thresh_sign', 0.25)
    iou_threshold = st.session_state.get('iou_thresh_sign', 0.45)
    
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"], key="video_uploader")
    
    # Initialize variables outside try block for finally
    cap = None
    temp_file_path = None
    
    try:
        if uploaded_file is not None:
            # Create a temporary file with proper extension
            file_ext = os.path.splitext(uploaded_file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tfile:
                tfile.write(uploaded_file.getbuffer())
                temp_file_path = tfile.name
            
            # Open video file
            cap = cv2.VideoCapture(temp_file_path)
            
            if not cap.isOpened():
                st.error("Error: Could not open video file.")
                return
                
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 30  # Default FPS if not available
            
            # Create placeholders
            frame_placeholder = st.empty()
            stop_button = st.button("Stop Video")
            
            while cap.isOpened() and not stop_button:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Process frame with current thresholds
                processed_frame = process_frame(frame, conf_threshold, iou_threshold)
                
                # Display the processed frame
                frame_placeholder.image(processed_frame, channels="BGR")
                
                # Add a small delay to control playback speed
                time.sleep(1.0 / fps)
                
                # Check if the stop button was pressed
                if stop_button:
                    break
    
    except Exception as e:
        st.error(f"An error occurred while processing the video: {str(e)}")
    
    finally:
        # Release resources
        if cap is not None:
            cap.release()
        
        # Clean up temporary file if it exists
        if temp_file_path is not None and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                st.warning(f"Could not delete temporary file: {str(e)}")

if __name__ == "__main__":
    main()
