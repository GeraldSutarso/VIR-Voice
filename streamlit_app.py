import streamlit as st
import torch
import whisper
import tempfile
import os
import speech_recognition as sr
import langid
import unidecode
from transformers import pipeline
import queue
import sounddevice as sd
import soundfile as sf
import numpy as np
import time
import threading
from streamlit_autorefresh import st_autorefresh  # pip install streamlit-autorefresh

# Ensure an asyncio event loop is running (if needed)
import asyncio
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# -------------------------------------------
# Print basic Torch info for debugging:
st.write("Torch version:", torch.__version__)
st.write("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    st.write("GPU:", torch.cuda.get_device_name(0))
else:
    st.write("No GPU detected - running on CPU.")

# -------------------------------------------
# Global Variables
global_audio_queue = queue.Queue()
current_rms = 0  # For audio level
recording_flag = [False]  # Use a mutable list for a global flag

# -------------------------------------------
# Load Models
device = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_whisper_model():
    # Use a smaller model if needed; using "large" is heavy
    return whisper.load_model("small", device=device)

model = load_whisper_model()

@st.cache_resource
def load_translator():
    return pipeline("translation", model="Helsinki-NLP/opus-mt-mul-en")

translator = load_translator()

# -------------------------------------------
# Microphone Device Selection
st.sidebar.markdown("### Microphone Settings")
devices = sd.query_devices()
input_devices = [(i, d["name"]) for i, d in enumerate(devices) if d["max_input_channels"] > 0]
if input_devices:
    mic_options = {f"{name} (Index {i})": i for i, name in input_devices}
    selected_mic = st.sidebar.selectbox("Select Microphone", options=list(mic_options.keys()))
    selected_device_index = mic_options[selected_mic]
else:
    st.sidebar.error("No microphone devices found!")
    selected_device_index = None

# -------------------------------------------
# Helper Functions

def transcribe_audio(audio_path):
    result = model.transcribe(audio_path)
    return result["text"]

def segment_and_colorize_text(text):
    """
    Segments the transcription using langid and assigns colors:
    - English: blue
    - Indonesian: red
    - Others: green
    """
    words = text.split()
    if not words:
        return ""
    segments = []
    current_lang, _ = langid.classify(words[0])
    current_segment = [words[0]]
    for word in words[1:]:
        lang, _ = langid.classify(word)
        if lang == current_lang:
            current_segment.append(word)
        else:
            segments.append((current_lang, " ".join(current_segment)))
            current_segment = [word]
            current_lang = lang
    segments.append((current_lang, " ".join(current_segment)))
    
    color_map = {"en": "blue", "id": "red"}
    default_color = "green"
    colored_output = ""
    for lang, segment in segments:
        color = color_map.get(lang, default_color)
        romanized = unidecode.unidecode(segment)
        colored_output += f'<span style="color:{color};">{romanized}</span> '
    return f"<div style='font-size:20px;'>{colored_output}</div>"

def translate_text(text):
    translation = translator(text, max_length=512)
    return translation[0]["translation_text"]

# -------------------------------------------
# Microphone Audio Callback
def audio_callback(indata, frames, time_info, status):
    global current_rms
    if status:
        print(status)
    current_rms = np.sqrt(np.mean(indata**2))
    global_audio_queue.put(indata.copy())

# -------------------------------------------
# Live Audio Level Indicator (Text-based)
def update_audio_level(placeholder, q, rec_flag):
    while rec_flag()[0]:
        frames = []
        while not q.empty():
            try:
                frames.append(q.get_nowait())
            except queue.Empty:
                break
        if frames:
            audio_data = np.concatenate(frames, axis=0)
            rms = np.sqrt(np.mean(audio_data**2))
            level = min(int(rms * 1000), 100)
            placeholder.text(f"Audio level: {level}")
        else:
            placeholder.text("Audio level: 0")
        time.sleep(0.5)
    placeholder.empty()

# -------------------------------------------
# Streamlit UI Layout

st.title("🎙️ Mixed-Language Speech Transcription & Translation")
st.write("Upload an audio file or record live using your microphone. The transcription will be segmented and color-coded based on language (English in blue, Indonesian in red, others in green), and a translation to English is provided below.")

# File Upload Section
st.markdown("## Upload Audio File")
uploaded_file = st.file_uploader("Choose an audio file (MP3, WAV, M4A)", type=["mp3", "wav", "m4a"])
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_audio_path = temp_file.name
    with st.spinner("Transcribing uploaded audio..."):
        transcription = transcribe_audio(temp_audio_path)
    st.markdown("### Transcription (Color-Coded):")
    st.markdown(segment_and_colorize_text(transcription), unsafe_allow_html=True)
    with st.spinner("Translating to English..."):
        translation = translate_text(transcription)
    st.markdown("### Translation to English:")
    st.write(translation)
    os.remove(temp_audio_path)

# Live Microphone Recording Section
st.markdown("## Live Microphone Recording")
audio_level_placeholder = st.empty()

col1, col2 = st.columns(2)
with col1:
    if st.button("Start Recording"):
        if selected_device_index is None:
            st.error("No microphone device selected!")
        else:
            recording_flag[0] = True
            while not global_audio_queue.empty():
                global_audio_queue.get()
            try:
                st.session_state.stream = sd.InputStream(
                    samplerate=8000,  # Lower sample rate to reduce data volume
                    channels=1,
                    device=selected_device_index,
                    callback=audio_callback
                )
                st.session_state.stream.start()
                st.success("Recording started. Speak now!")
                threading.Thread(target=update_audio_level, args=(audio_level_placeholder, global_audio_queue, lambda: recording_flag), daemon=True).start()
            except Exception as e:
                st.error(f"Error starting recording: {e}")
with col2:
    if st.button("Stop Recording"):
        if not recording_flag[0]:
            st.warning("Recording is not active.")
        else:
            recording_flag[0] = False
            try:
                st.session_state.stream.stop()
                st.session_state.stream.close()
            except Exception as e:
                st.error(f"Error stopping recording: {e}")
            audio_level_placeholder.empty()
            frames = []
            while not global_audio_queue.empty():
                frames.append(global_audio_queue.get())
            if frames:
                audio_data = np.concatenate(frames, axis=0)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                    temp_filename = temp_file.name
                sf.write(temp_filename, audio_data, 8000)
                with st.spinner("Transcribing recorded audio..."):
                    transcription = transcribe_audio(temp_filename)
                st.markdown("### Transcription (Color-Coded):")
                st.markdown(segment_and_colorize_text(transcription), unsafe_allow_html=True)
                with st.spinner("Translating to English..."):
                    translation = translate_text(transcription)
                st.markdown("### Translation to English:")
                st.write(translation)
                os.remove(temp_filename)
            else:
                st.error("No audio data recorded.")

# Auto-refresh if recording is active (to update audio level indicator)
if recording_flag[0]:
    st_autorefresh(interval=1000, limit=1000, key="audio_level_autorefresh")

st.markdown("---")
st.markdown("🛠 Powered by **OpenAI Whisper**, **Streamlit**, **Langid**, **Unidecode**, and **Transformers**")
