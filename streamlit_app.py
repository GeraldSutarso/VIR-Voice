
import streamlit as st
import numpy as np
import librosa
import tempfile
import plotly.express as px
import tensorflow as tf
from tensorflow.keras.models import load_model

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="🎭",
    layout="wide"
)

# --- CUSTOM STYLES ---
st.markdown("""
<style>
    .title { font-size: 2.5rem; color: #1E88E5; text-align: center; }
    .sub-title { font-size: 1.5rem; color: #424242; }
    .result-box { font-size: 1.8rem; font-weight: bold; text-align: center; }
    .stAudio { margin-top: 1rem; margin-bottom: 1rem; }
</style>
""", unsafe_allow_html=True)

# --- MANUALLY DEFINED EMOTIONS ---
EMOTION_LABELS = ["Neutral", "Calm", "Happy", "Sad", "Angry", "Fearful", "Disgust", "Surprised"]

# --- LOAD MODEL ---
@st.cache_resource
def load_emotion_model():
    try:
        model = load_model("emotion_recognition_lstm_finetune.h5")
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

# --- FEATURE EXTRACTION ---
def extract_features(file_path, max_pad_len=100):
    audio, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)

    pad_width = max_pad_len - mfcc.shape[1]
    if pad_width > 0:
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_pad_len]

    return mfcc.T

# --- PREDICTION FUNCTION ---
def predict_emotion(model, audio_file):
    features = extract_features(audio_file)
    features = np.expand_dims(features, axis=0)  # Add batch dimension

    predictions = model.predict(features)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = predictions[0][predicted_class]

    emotion = EMOTION_LABELS[predicted_class]
    all_emotions = {EMOTION_LABELS[i]: predictions[0][i] for i in range(len(predictions[0]))}

    return emotion, confidence, all_emotions

# --- PLOT WAVEFORM ---
def plot_waveform(audio_file):
    y, sr = librosa.load(audio_file)
    fig = px.line(x=np.arange(len(y)) / sr, y=y, labels={'x': 'Time (s)', 'y': 'Amplitude'}, title='Audio Waveform')
    fig.update_layout(height=250)
    return fig

# --- MAIN APP ---
def main():
    st.markdown("<h1 class='title'>Speech Emotion Recognition</h1>", unsafe_allow_html=True)
    
    # Sidebar Info
    st.sidebar.title("Instructions")
    st.sidebar.info("Upload a WAV file to analyze its emotional content. The model detects emotions like Neutral, Happy, Sad, Angry, etc.")
    
    # Load Model
    model = load_emotion_model()
    if model is None:
        return

    # File Upload
    st.markdown("<h2 class='sub-title'>Upload Audio File</h2>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a WAV file", type=["wav"])

    # Process File
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            audio_path = tmp_file.name

        # Display Audio Player
        st.audio(uploaded_file)

        # Display Waveform
        waveform_fig = plot_waveform(audio_path)
        st.plotly_chart(waveform_fig, use_container_width=True)

        # Analyze Button
        if st.button("Analyze Emotion"):
            with st.spinner("Analyzing..."):
                emotion, confidence, all_emotions = predict_emotion(model, audio_path)

                # Display Result
                st.markdown(f"<h2 class='sub-title'>Detected Emotion</h2>", unsafe_allow_html=True)
                st.markdown(f"<div class='result-box'>{emotion} ({confidence:.2f})</div>", unsafe_allow_html=True)

                # Display Confidence Scores
                st.markdown("#### Emotion Confidence Scores")
                emotion_fig = px.bar(x=list(all_emotions.keys()), y=list(all_emotions.values()), 
                                     labels={'x': 'Emotion', 'y': 'Confidence'},
                                     title='Emotion Prediction Confidence Scores',
                                     color=list(all_emotions.values()),
                                     color_continuous_scale='Blues')
                st.plotly_chart(emotion_fig, use_container_width=True)

if __name__ == "__main__":
    main()
