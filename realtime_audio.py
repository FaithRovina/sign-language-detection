import streamlit as st
import numpy as np
import sounddevice as sd
import queue
import base64
import requests
import time

RATE = 16000
CHUNK = int(RATE / 4)  # 250ms

class RealTimeAudioStream:
    def __init__(self):
        self.q = queue.Queue()
        self.is_recording = False
        self.stream = None

    def callback(self, indata, frames, time_info, status):
        if self.is_recording:
            self.q.put(indata.copy())

    def start(self):
        self.is_recording = True
        self.stream = sd.InputStream(
            samplerate=RATE,
            channels=1,
            dtype='int16',
            callback=self.callback
        )
        self.stream.start()

    def stop(self):
        self.is_recording = False
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def get_audio_chunk(self):
        frames = []
        while not self.q.empty():
            frames.append(self.q.get())
        if frames:
            return np.concatenate(frames, axis=0)
        return None

def transcribe_chunk(chunk):
    api_key = st.secrets["GOOGLE_SPEECH_API_KEY"]
    audio_content = base64.b64encode(chunk.tobytes()).decode("utf-8")
    url = f"https://speech.googleapis.com/v1/speech:recognize?key={api_key}"
    data = {
        "config": {
            "encoding": "LINEAR16",
            "sampleRateHertz": RATE,
            "languageCode": "en-US",
            "enableAutomaticPunctuation": True
        },
        "audio": {
            "content": audio_content
        }
    }
    try:
        response = requests.post(url, json=data)
        result = response.json()
        if "results" in result:
            return " ".join([alt.get("transcript", "") for r in result["results"] for alt in r.get("alternatives", [])])
        elif "error" in result:
            st.error(f"API Error: {result['error'].get('message', 'Unknown error')}")
        return ""
    except Exception as e:
        st.error(f"Transcription error: {str(e)}")
        return ""

def run_realtime_transcription():
    st.write("Press start to see live transcription. Speak into your mic.")
    if 'rt_audio' not in st.session_state:
        st.session_state.rt_audio = RealTimeAudioStream()
    if 'rt_transcript' not in st.session_state:
        st.session_state.rt_transcript = ""
    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ Start Live Transcription", key="rt_start"):
            st.session_state.rt_audio.start()
            st.session_state.rt_transcript = ""
            st.session_state.rt_running = True
    with col2:
        if st.button("⏹️ Stop Live Transcription", key="rt_stop"):
            st.session_state.rt_audio.stop()
            st.session_state.rt_running = False
    if st.session_state.get('rt_running', False):
        placeholder = st.empty()
        st.markdown('<span style="color: green; font-weight: bold; font-size: 1.2em;">🟢 Live transcription is running...</span>', unsafe_allow_html=True)
        # Create the text_area ONCE before the loop and only update its value
        text_area = placeholder.text_area("Live Transcription", value=st.session_state.rt_transcript, height=350, key="live_transcription_box_rt")
        while st.session_state.get('rt_running', False):
            chunk = st.session_state.rt_audio.get_audio_chunk()
            if chunk is not None and len(chunk) > 0:
                text = transcribe_chunk(chunk)
                if text:
                    st.session_state.rt_transcript += " " + text
                    st.experimental_rerun()
                # The text_area is only created once per rerun; its value will update automatically via st.session_state.
            time.sleep(0.5)
        # The text_area is only created once per rerun; its value will update automatically via st.session_state.
