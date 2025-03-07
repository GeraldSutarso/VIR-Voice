import streamlit as st
import whisper
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
import tempfile
import os

# Load the model (CPU only)
@st.cache_resource
def load_model():
    return whisper.load_model("small")

model = load_model()

# App title
st.set_page_config(page_title="Speech Recognition App", layout="wide")

# Sidebar - Language Selection
st.sidebar.title("Settings")
language = st.sidebar.selectbox("Choose language:", ["English", "Indonesian"])

# Define Whisper language codes
language_codes = {"English": "en", "Indonesian": "id"}

# Tabs for navigation
tab1, tab2, tab3 = st.tabs(["📖 Introduction", "📂 Upload File", "🎤 Live Mic"])

# **TAB 1: Introduction**
with tab1:
    st.title("Speech Recognition App 🎙️")
    st.write("This app transcribes speech using OpenAI Whisper.")

    st.subheader("Features:")
    st.markdown("- **Upload an audio file** for transcription.")
    st.markdown("- **Use live microphone input** for real-time speech recognition.")
    st.markdown("- **Supports multiple languages** (English & Indonesian).")

# **TAB 2: Upload File for Transcription**
with tab2:
    st.title("Upload Audio File")
    uploaded_file = st.file_uploader("Choose an audio file (WAV, MP3, etc.)", type=["wav", "mp3", "ogg"])

    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav")

        # Save the file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(uploaded_file.read())
            temp_audio_path = temp_audio.name

        # Run Whisper on the file
        with st.spinner("Transcribing..."):
            result = model.transcribe(temp_audio_path, language=language_codes[language])

        st.subheader("Transcription Result:")
        st.write(result["text"])

        # Delete temp file
        os.remove(temp_audio_path)

# **TAB 3: Live Microphone Transcription**
with tab3:
    st.title("Live Microphone Recording")

    duration = st.slider("Select recording duration (seconds)", min_value=1, max_value=10, value=5)

    if st.button("🎙️ Start Recording"):
        with st.spinner("Recording..."):
            sample_rate = 44100
            recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype=np.int16)
            sd.wait()

            # Save to a temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
                wav.write(temp_audio.name, sample_rate, recording)
                temp_audio_path = temp_audio.name

            st.audio(temp_audio_path, format="audio/wav")

            # Run Whisper on the recorded audio
            with st.spinner("Transcribing..."):
                result = model.transcribe(temp_audio_path, language=language_codes[language])

            st.subheader("Transcription Result:")
            st.write(result["text"])

            # Delete temp file
            os.remove(temp_audio_path)
