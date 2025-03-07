import streamlit as st
import torch
import whisper
import tempfile
import os
import langid
import unidecode
from transformers import pipeline
import numpy as np
import pandas as pd
import random
import pycountry
import asyncio

# Ensure an asyncio event loop is running
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# -------------------------------------------
# Debug Info
st.write("Torch version:", torch.__version__)
st.write("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    st.write("GPU:", torch.cuda.get_device_name(0))
else:
    st.write("No GPU detected - running on CPU.")

# -------------------------------------------
# Load Models
device = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_whisper_model():
    return whisper.load_model("large", device=device)

model = load_whisper_model()

@st.cache_resource
def load_translator():
    return pipeline("translation", model="Helsinki-NLP/opus-mt-mul-en")

translator = load_translator()

# -------------------------------------------
# Helper Functions

def get_language_name(lang_code):
    """
    Returns the full language name for a given ISO language code.
    If not found, it returns the code itself.
    """
    try:
        language = pycountry.languages.get(alpha_2=lang_code)
        if language is not None:
            return language.name
    except Exception:
        pass
    return lang_code

def transcribe_audio(audio_path):
    result = model.transcribe(audio_path)
    return result["text"]

def transcribe_audio_word_level(audio_path):
    """
    Transcribes the audio file and returns a list of dictionaries,
    each containing a word, its start/end timestamps, and an improved detected language.
    
    This version leverages the overall segment language as context. For each segment,
    we detect the language for the entire segment text. Then for each word, if its individual
    detection (using langid) returns English while the segment overall is Indonesian/Malay
    (or if the segment detection is stronger), we override the word's language.
    
    Note: word_timestamps is experimental and requires a compatible Whisper version.
    """
    result = model.transcribe(audio_path, word_timestamps=True)
    word_details = []
    for segment in result.get("segments", []):
        # Get overall segment language detection (using the full segment text)
        segment_text = segment.get("text", "")
        seg_lang, seg_conf = langid.classify(segment_text)
        
        if "words" in segment:
            for word_info in segment["words"]:
                word_text = word_info["word"].strip()
                start = word_info["start"]
                end = word_info["end"]
                # Detect language for the individual word
                word_lang, word_conf = langid.classify(word_text)
                # Heuristic:
                # If the word is classified as English but the segment as Indonesian/Malay,
                # or if the segment detection is notably stronger, override with the segment's language.
                if word_lang == "en" and seg_lang in ["id", "ms"]:
                    lang = seg_lang
                elif word_conf < seg_conf and seg_conf > 0.9:
                    lang = seg_lang
                else:
                    lang = word_lang
                word_details.append({"word": word_text, "start": start, "end": end, "language": lang})
        else:
            # Fallback: split the segment text and assign the segment's language to each word.
            words = segment_text.split()
            for word in words:
                start = segment["start"]
                end = segment["end"]
                lang = seg_lang  # assign the segment language
                word_details.append({"word": word, "start": start, "end": end, "language": lang})
    return word_details

def colored_transcription_from_word_details(word_details):
    """
    Returns an HTML string with each word color-coded based on detected language.
    Colors for each language are automatically assigned from a palette.
    A legend is added to indicate which color corresponds to which language (with full names).
    """
    # Define a palette of colors
    palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
               '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    color_map = {}
    colored_output = ""
    
    for wd in word_details:
        lang = wd["language"]
        if lang not in color_map:
            if palette:
                color_map[lang] = palette.pop(0)
            else:
                # If the palette is exhausted, generate a random color
                color_map[lang] = "#%06x" % random.randint(0, 0xFFFFFF)
        color = color_map[lang]
        colored_output += f'<span style="color:{color};">{wd["word"]}</span> '
    
    # Create a legend with full language names
    legend_items = [
        f"<span style='color:{clr}; font-weight:bold;'>{get_language_name(lang)}</span>"
        for lang, clr in color_map.items()
    ]
    legend = "<br><br><strong>Legend:</strong> " + ", ".join(legend_items)
    return f"<div style='font-size:20px;'>{colored_output}</div>{legend}"

def translate_text(text):
    translation = translator(text, max_length=512)
    return translation[0]["translation_text"]

# -------------------------------------------
# Main UI Layout

st.title("🎙️ Mixed-Language Speech Transcription & Translation")
st.write(
    "Upload an audio file. The transcription now uses improved word-level language detection by leveraging segment context. "
    "A translation to English is also provided below."
)

# --------- File Upload Section ---------
st.markdown("## Upload Audio File")
uploaded_file = st.file_uploader("Choose an audio file (MP3, WAV, M4A)", type=["mp3", "wav", "m4a"])
if uploaded_file is not None:
    # Save the uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_audio_path = temp_file.name

    with st.spinner("Transcribing uploaded audio with word-level detection..."):
        word_details = transcribe_audio_word_level(temp_audio_path)
    
    # Reconstruct the full transcription from the detected words
    full_transcription = " ".join([wd["word"] for wd in word_details])
    
    st.markdown("### Transcription (Word-level Detection and Color-Coded):")
    st.markdown(colored_transcription_from_word_details(word_details), unsafe_allow_html=True)
    
    st.markdown("#### Detailed Word Information:")
    df = pd.DataFrame(word_details)
    st.table(df)
    
    with st.spinner("Translating to English..."):
        translation = translate_text(full_transcription)
    st.markdown("### Translation to English:")
    st.write(translation)
    
    os.remove(temp_audio_path)
