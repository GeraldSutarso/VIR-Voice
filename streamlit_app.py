import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
from tempfile import NamedTemporaryFile
from pydub import AudioSegment
import io
import imageio_ffmpeg as ffmpeg
from pydub import AudioSegment

# Set the path for both the converter and ffprobe
AudioSegment.converter = ffmpeg.get_ffmpeg_exe()
# If needed, you can use the same binary or check the package docs for ffprobe:
AudioSegment.ffprobe = ffmpeg.get_ffmpeg_exe()


# Custom InputLayer to handle batch_shape -> batch_input_shape mapping
from tensorflow.keras.layers import InputLayer as BaseInputLayer

class CustomInputLayer(BaseInputLayer):
    def __init__(self, *args, **kwargs):
        # Rename 'batch_shape' to 'batch_input_shape' if present
        if 'batch_shape' in kwargs:
            kwargs['batch_input_shape'] = kwargs.pop('batch_shape')
        super().__init__(*args, **kwargs)

# Load pre-trained model using the custom InputLayer
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        'emotion_recognition_lstm.h5', 
        custom_objects={'InputLayer': CustomInputLayer}
    )

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

# Streamlit app UI
st.title("🎤 Voice Emotion Recognition")
st.write("Upload a WAV or MP3 file to analyze the emotional state")

# Allow both wav and mp3 file types
uploaded_file = st.file_uploader("Choose audio file", type=["wav", "mp3"])

if uploaded_file is not None:
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    # Convert MP3 to WAV if needed
    if file_extension == 'mp3':
        # Load the MP3 file with pydub
        audio = AudioSegment.from_file(uploaded_file, format="mp3")
        wav_io = io.BytesIO()
        audio.export(wav_io, format="wav")
        wav_io.seek(0)
        audio_data = wav_io.read()
        
        with NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_data)
            tmp_path = tmp_file.name
    else:
        # For WAV, save directly
        with NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
    
    # Display audio player
    st.audio(uploaded_file, format=f'audio/{file_extension}')
    
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
