import streamlit as st
import queue
import sounddevice as sd
import torch
import numpy as np
import whisper
import threading
import tempfile
import os
import wave

# Check if CUDA is available and load Whisper Large model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = whisper.load_model("medium", device=device)

# Initialize global variables for streaming
st.session_state["recording"] = False
audio_queue = queue.Queue()


# Function to capture microphone input
def audio_callback(indata, frames, time, status):
    """Callback function to store recorded audio chunks into the queue."""
    if status:
        print(status)
    audio_queue.put(indata.copy())


# Function to start recording
def start_recording():
    """Starts recording and saves audio into a temporary file."""
    st.session_state["recording"] = True
    audio_queue.queue.clear()  # Clear previous audio
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio_path = temp_audio.name

    def record_audio():
        """Handles real-time audio recording in a separate thread."""
        samplerate = 16000  # Whisper works best with 16kHz audio
        channels = 1
        dtype = np.int16

        with wave.open(temp_audio_path, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2)  # 16-bit audio
            wf.setframerate(samplerate)

            with sd.InputStream(callback=audio_callback, samplerate=samplerate, channels=channels, dtype=dtype):
                while st.session_state["recording"]:
                    try:
                        audio_chunk = audio_queue.get(timeout=1)
                        wf.writeframes(audio_chunk)
                    except queue.Empty:
                        pass  # No data in queue

    recording_thread = threading.Thread(target=record_audio)
    recording_thread.start()
    st.session_state["temp_audio_path"] = temp_audio_path


# Function to stop recording
def stop_recording():
    """Stops recording and processes the recorded audio."""
    st.session_state["recording"] = False

    # Wait for the queue to process remaining audio
    while not audio_queue.empty():
        pass

    # Load and transcribe audio
    temp_audio_path = st.session_state.get("temp_audio_path", None)
    if temp_audio_path and os.path.exists(temp_audio_path):
        st.write("🔍 Transcribing...")
        result = model.transcribe(temp_audio_path)
        st.write("📝 Transcription:", result["text"])
        os.remove(temp_audio_path)  # Clean up temp file


# Streamlit UI
st.title("🎙️ Real-time Speech-to-Text with Whisper")

# Display buttons
col1, col2 = st.columns(2)
with col1:
    if st.button("Start Recording 🎤", key="start"):
        if not st.session_state["recording"]:
            start_recording()

with col2:
    if st.button("Stop Recording ⏹️", key="stop"):
        if st.session_state["recording"]:
            stop_recording()
