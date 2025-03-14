import streamlit as st
import pickle
import numpy as np
import librosa
import os
import tempfile
import matplotlib.pyplot as plt
import pandas as pd
import time
from scipy.io import wavfile
import plotly.express as px
import plotly.graph_objects as go
import tensorflow as tf
from tensorflow.keras.models import load_model

# Set page config
st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="🎭",
    layout="wide"
)

# Define CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
    }
    .result-text {
        font-size: 1.8rem;
        font-weight: bold;
    }
    .stAudio {
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Function to extract features (same as in original code)
@st.cache_data
def extract_features(file_path, max_pad_len=100):
    audio, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)

    pad_width = max_pad_len - mfcc.shape[1]
    if pad_width > 0:
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_pad_len]

    return mfcc.T

# Important: Re-define the EmotionEnsembleModel class here so it's available during unpickling
class EmotionEnsembleModel:
    def __init__(self, models, label_encoder):
        self.models = models
        self.label_encoder = label_encoder
        
    def predict(self, audio_file, max_pad_len=100):
        # Extract features using the same function as in training
        features = extract_features(audio_file, max_pad_len)
        features = np.expand_dims(features, axis=0)
        
        # Get predictions from all models
        predictions = [model.predict(features) for model in self.models]
        
        # Average predictions
        ensemble_pred = np.mean(predictions, axis=0)
        predicted_class = np.argmax(ensemble_pred, axis=1)[0]
        
        # Map back to emotion label
        emotion = self.label_encoder.inverse_transform([predicted_class])[0]
        confidence = ensemble_pred[0][predicted_class]
        
        return emotion, confidence, {self.label_encoder.inverse_transform([i])[0]: ensemble_pred[0][i] for i in range(len(ensemble_pred[0]))}

# Function to visualize the audio waveform
def plot_waveform(audio_file):
    y, sr = librosa.load(audio_file)
    fig = px.line(x=np.arange(len(y))/sr, y=y, 
                  labels={'x': 'Time (s)', 'y': 'Amplitude'},
                  title='Audio Waveform')
    fig.update_layout(height=250)
    return fig

# Function to visualize the emotion predictions
def plot_emotion_predictions(predictions):
    emotions = list(predictions.keys())
    probs = list(predictions.values())
    
    # Create sorted data for the bar chart
    sorted_data = sorted(zip(emotions, probs), key=lambda x: x[1], reverse=True)
    sorted_emotions, sorted_probs = zip(*sorted_data)
    
    fig = px.bar(
        x=sorted_emotions, 
        y=sorted_probs,
        labels={'x': 'Emotion', 'y': 'Confidence'},
        title='Emotion Prediction Confidence Scores',
        color=sorted_probs,
        color_continuous_scale='Blues'
    )
    fig.update_layout(height=400)
    return fig

# Alternative model loading approach
# def load_model_manually():
#     try:
#         # Load label encoder
#         with open('label_encoder.pkl', 'rb') as f:
#             label_encoder = pickle.load(f)
        
#         # Load individual models
#         model1 = load_model("emotion_recognition_lstm.h5")
#         model2 = load_model("emotion_model_2.h5")
#         model3 = load_model("emotion_model_3.h5")
        
#         # Create ensemble model instance
#         ensemble_model = EmotionEnsembleModel(
#             models=[model1, model2, model3],
#             label_encoder=label_encoder
#         )
        
#         return ensemble_model
#     except FileNotFoundError as e:
#         st.error(f"Model file not found: {e}. Please ensure all model files are in the same directory as this app.")
#         return None

# Alternative model loading approach
def load_model_manually():
    try:
        # Load label encoder
        with open('label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        
        # Load the new model
        model = load_model("emotion_recognition_lstm_no_overfit.h5")
        
        return model, label_encoder
    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}. Please ensure the model file is in the same directory as this app.")
        return None, None


# Load the ensemble model
@st.cache_resource
def load_model():
    # try:
    #     # First try loading the pickled ensemble model
    #     with open('emotion_ensemble_model.pkl', 'rb') as f:
    #         ensemble_model = pickle.load(f)
    #     return ensemble_model
    # except (FileNotFoundError, AttributeError):
    #     st.warning("Could not load the ensemble model directly. Trying to load individual models...")
    #     # Fall back to loading models manually
        return load_model_manually()

# Main app logic
def main():

    st.sidebar.title("About")
    st.sidebar.info(
        "This app uses a deep learning ensemble model to recognize emotions in speech. "
        "The model was trained on the RAVDESS emotional speech audio dataset and can "
        "identify 8 different emotions: Neutral, Calm, Happy, Sad, Angry, Fearful, "
        "Disgust, and Surprised."
    )
    
    st.sidebar.title("Instructions")
    st.sidebar.markdown(
        """
        1. Upload an audio file (.wav format)
        2. Wait for the analysis to complete
        3. View the predicted emotion and confidence scores
        
        For best results, use:
        - Clear audio with minimal background noise
        - Speech with emotional expression
        - Short audio clips (2-10 seconds)
        """
    )
    
    # Load the model
    model = load_model()
    
    if model is None:
        st.error("Could not load the required model files. Please check the setup instructions.")
        return
    
    # File uploader
    st.markdown("<h2 class='sub-header'>Upload Audio</h2>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a WAV file", type=["wav"])
    
    # Sample audio option
    st.markdown("Or try a sample:")
    sample_options = ["None", "Happy Sample", "Sad Sample", "Angry Sample"]
    sample_choice = st.selectbox("Select sample audio", sample_options)
    
    sample_file = None
    if sample_choice != "None":
        # This is a placeholder. In a real app, you would have these sample files
        sample_paths = {
            "Happy Sample": "sample_happy.wav",
            "Sad Sample": "sample_sad.wav",
            "Angry Sample": "sample_angry.wav"
        }
        
        if os.path.exists(sample_paths.get(sample_choice, "")):
            sample_file = sample_paths.get(sample_choice)
            st.audio(sample_file)
        else:
            st.warning(f"Sample file {sample_paths.get(sample_choice)} not found.")
    
    # Analyze button
    if uploaded_file is not None or sample_file is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<h2 class='sub-header'>Audio Input</h2>", unsafe_allow_html=True)
            
            # Save uploaded file to temp file
            if uploaded_file is not None:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    audio_path = tmp_file.name
                
                # Display the audio
                st.audio(uploaded_file)
                
                # Plot waveform
                waveform_fig = plot_waveform(audio_path)
                st.plotly_chart(waveform_fig, use_container_width=True)
            
            elif sample_file is not None:
                audio_path = sample_file
                
                # Plot waveform
                waveform_fig = plot_waveform(audio_path)
                st.plotly_chart(waveform_fig, use_container_width=True)
        
        # Analysis button
        analyze_button = st.button("Analyze Emotion")
        
        if analyze_button:
            with st.spinner('Analyzing audio...'):
                # Create a progress bar
                progress_bar = st.progress(0)
                
                # Simulate processing steps
                progress_bar.progress(25)
                time.sleep(0.5)  # Feature extraction
                
                try:
                    # Predict emotion
                    # emotion, confidence, all_emotions = model.predict(audio_path)

                    # Load extracted features
                    features = extract_features(audio_path)
                    features = np.expand_dims(features, axis=0)

                    # Get predictions
                    predictions = model.predict(features)
                    predicted_class = np.argmax(predictions, axis=1)[0]

                    # Map back to emotion label
                    emotion = label_encoder.inverse_transform([predicted_class])[0]
                    confidence = predictions[0][predicted_class]

                    # Store all emotion confidence scores
                    all_emotions = {label_encoder.inverse_transform([i])[0]: predictions[0][i] for i in range(len(predictions[0]))}

                    
                    progress_bar.progress(50)
                    time.sleep(0.5)  # Processing
                    
                    progress_bar.progress(75)
                    time.sleep(0.5)  # Finalizing
                    
                    progress_bar.progress(100)
                    
                    with col2:
                        st.markdown("<h2 class='sub-header'>Analysis Results</h2>", unsafe_allow_html=True)
                        
                        # Display the primary emotion with a nice colored box
                        emotion_colors = {
                            "Neutral": "#9E9E9E",
                            "Calm": "#64B5F6",
                            "Happy": "#FFD54F",
                            "Sad": "#90CAF9",
                            "Angry": "#EF5350",
                            "Fearful": "#CE93D8",
                            "Disgust": "#81C784",
                            "Surprised": "#FFB74D"
                        }
                        
                        emotion_color = emotion_colors.get(emotion, "#9E9E9E")
                        
                        st.markdown(
                            f"""
                            <div style="background-color: {emotion_color}; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
                                <h3 style="color: white; margin: 0;">Detected Emotion: 
                                    <span class="result-text">{emotion}</span>
                                </h3>
                                <p style="color: white; margin: 0;">Confidence: {confidence:.2f}</p>
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
                        
                        # Display all emotion probabilities
                        st.markdown("#### Emotion Confidence Scores")
                        
                        # Create and display the emotions plot
                        emotions_fig = plot_emotion_predictions(all_emotions)
                        st.plotly_chart(emotions_fig, use_container_width=True)
                        
                        # Add explanation
                        st.markdown("""
                        **Understanding the results:**
                        
                        The model predicts the likelihood of each emotion in the audio. The emotion with the highest score is considered the primary detected emotion. Higher confidence scores indicate stronger emotional signals in the speech.
                        """)
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
                    st.info("Check that the audio file is in the correct format (WAV) and contains clear speech.")
            
            # Clean up temp file
            if uploaded_file is not None:
                os.unlink(audio_path)

    st.markdown("---")

    
if __name__ == "__main__":
    main()