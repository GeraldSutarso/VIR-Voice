import streamlit as st
import numpy as np
import librosa
import librosa.display
import tensorflow as tf
import matplotlib.pyplot as plt
import os

# Load the trained model
MODEL_PATH = "fine_tuned_emotion_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Define emotion labels
emotion_labels = {
    0: "Angry",
    1: "Calm",
    2: "Happy",
    3: "Sad",
    4: "Fearful",
    5: "Disgust",
    6: "Surprised",
    7: "Neutral"
}

# Function to extract MFCC features
def extract_features(file_path, max_pad_len=100):
    audio, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    
    pad_width = max_pad_len - mfcc.shape[1]
    if pad_width > 0:
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_pad_len]
    
    return mfcc.T

# Streamlit UI
st.title("🎙️ Emotion Recognition from Audio")
st.write("Upload an audio file and the model will predict the emotion.")

# Upload audio file
uploaded_file = st.file_uploader("Choose an audio file...", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")

    # Save the uploaded file
    file_path = "temp_audio.wav"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Extract features
    features = extract_features(file_path)
    features = np.expand_dims(features, axis=0)  # Reshape for model input

    # Predict emotion
    prediction = model.predict(features)
    predicted_class = np.argmax(prediction)
    predicted_emotion = emotion_labels[predicted_class]

    # Display result
    st.subheader("Predicted Emotion:")
    st.write(f"🎭 **{predicted_emotion}**")

    # Plot MFCC Features
    st.subheader("MFCC Features")
    fig, ax = plt.subplots(figsize=(8, 4))
    librosa.display.specshow(features[0].T, x_axis="time", cmap="viridis")
    plt.colorbar()
    st.pyplot(fig)

    # Remove temp file
    os.remove(file_path)
