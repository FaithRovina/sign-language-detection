import streamlit as st
import asyncio
import base64
import json
import pyaudio
import os
from pathlib import Path

def run_realtime_transcription():
    # Session state
    if 'text' not in st.session_state:
        st.session_state['text'] = 'Listening...'
        st.session_state['run'] = False

    # Audio parameters 
    st.sidebar.header('Audio Parameters')
    FRAMES_PER_BUFFER = int(st.sidebar.text_input('Frames per buffer', 3200))
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = int(st.sidebar.text_input('Rate', 16000))
    p = pyaudio.PyAudio()

    # Open an audio stream with above parameter settings
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=FRAMES_PER_BUFFER
    )

    # Start/stop audio transmission
    def start_listening():
        st.session_state['run'] = True

    def download_transcription():
        read_txt = open('transcription.txt', 'r')
        st.download_button(
            label="Download transcription",
            data=read_txt,
            file_name='transcription_output.txt',
            mime='text/plain')

    def stop_listening():
        st.session_state['run'] = False

    # Web user interface
    st.markdown('### 🎙️ Real-Time Transcription')

    if st.session_state['run']:
        # Real-time transcription logic removed due to websocket removal.
        # Placeholder for future implementation.
        st.info('Real-time transcription is currently unavailable in this version.')

    # Always display transcript area
    st.text_area("Live Transcription", value=st.session_state['text'], height=350)

    if Path('transcription.txt').is_file():
        st.markdown('### Download')
        download_transcription()
        os.remove('transcription.txt')

   