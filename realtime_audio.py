import streamlit as st
import websockets
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
        async def send_receive():
            URL = f"wss://api.assemblyai.com/v2/realtime/ws?sample_rate={RATE}"
            try:
                async with websockets.connect(
                    URL,
                    additional_headers={'Authorization': st.secrets['api_key']},
                    ping_interval=5,
                    ping_timeout=20
                ) as _ws:
                    await asyncio.sleep(0.1)
                    await _ws.recv()
                    async def send():
                        while st.session_state['run']:
                            try:
                                data = stream.read(FRAMES_PER_BUFFER)
                                data = base64.b64encode(data).decode("utf-8")
                                json_data = json.dumps({"audio_data":str(data)})
                                await _ws.send(json_data)
                            except Exception as e:
                                break
                            await asyncio.sleep(0.01)
                    async def receive():
                        while st.session_state['run']:
                            try:
                                result_str = await _ws.recv()
                                result = json.loads(result_str)['text']
                                if json.loads(result_str)['message_type']=='FinalTranscript':
                                    st.session_state['text'] = result
                                    with open('transcription.txt', 'a') as transcription_txt:
                                        transcription_txt.write(st.session_state['text'] + ' ')
                            except Exception as e:
                                break
                    await asyncio.gather(send(), receive())
            except Exception as e:
                st.error(f"WebSocket error: {e}")
        asyncio.run(send_receive())

    # Always display transcript area
    st.text_area("Live Transcription", value=st.session_state['text'], height=350)

    if Path('transcription.txt').is_file():
        st.markdown('### Download')
        download_transcription()
        os.remove('transcription.txt')

   