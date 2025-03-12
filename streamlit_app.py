import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
from tempfile import NamedTemporaryFile

# Load pre-trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('emotion_recognition_lstm.h5')

model = load_model()

# Emotion labels (ensure order matches training labels)
emotion_labels = [
    'Neutral', 'Calm', 'Happy', 'Sad', 
    'Angry', 'Fearful', 'Disgust', 'Surprised'
]

# Feature extraction function
def extract_features(audio_path, max_pad_len=100):
    audio, sr = librosa.load(audio_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    
    # Pad/truncate to fixed length
    if mfcc.shape[1] < max_pad_len:
        mfcc = np.pad(mfcc, ((0, 0), (0, max_pad_len - mfcc.shape[1])), mode='constant')
    else:
        mfcc = mfcc[:, :max_pad_len]
    
    return mfcc.T

# Streamlit app
st.title("🎤 Voice Emotion Recognition")
st.write("Upload a WAV file to analyze emotional state")

uploaded_file = st.file_uploader("Choose audio file", type=["wav"])

if uploaded_file is not None:
    # Save uploaded file temporarily
    with NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    
    # Display audio player
    st.audio(uploaded_file, format='audio/wav')
    
    # Process audio
    try:
        features = extract_features(tmp_path)
        input_data = np.expand_dims(features, axis=0)  # Add batch dimension
        
        # Make prediction
        prediction = model.predict(input_data)
        predicted_class = np.argmax(prediction[0])
        confidence = np.max(prediction[0])
        
        # Display results
        st.subheader("Analysis Results")
        emoji_dict = {
            'Neutral': "😐",
            'Calm': "😌",
            'Happy': "😄",
            'Sad': "😢",
            'Angry': "😠",
            'Fearful': "😨",
            'Disgust': "🤢",
            'Surprised': "😲"
        }
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Predicted Emotion", 
                     f"{emotion_labels[predicted_class]} {emoji_dict[emotion_labels[predicted_class]]}")
        with col2:
            st.metric("Confidence", f"{confidence:.2%}")
            
        # Show probability distribution
        st.write("### Emotion Probability Distribution")
        probs = {emotion: float(prediction[0][i]) for i, emotion in enumerate(emotion_labels)}
        st.bar_chart(probs)
        
    except Exception as e:
        st.error(f"Error processing audio: {str(e)}")