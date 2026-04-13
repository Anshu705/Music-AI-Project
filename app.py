import streamlit as st
import librosa
import numpy as np
import keras
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

st.set_page_config(page_title="Manan's Music AI", page_icon="🎵")

@st.cache_resource
def load_ai():
    model = keras.models.load_model('music_mood_model_104.keras')
    data = pd.read_csv('music_database_104.csv')
    X_raw = data[['BPM', 'MFCC_Mean', 'Brightness']].values
    scaler = StandardScaler().fit(X_raw)
    encoder = LabelEncoder()
    encoder.fit(data['Mood'])
    return model, scaler, encoder

model, scaler, encoder = load_ai()

# --- THE MISSING PREDICT_MOOD FUNCTION ---
def predict_mood(file_path):
    # 1. Load Audio
    y, sr = librosa.load(file_path, duration=30)
    
    # 2. Extract the SAME 7 FEATURES as training
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13))
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    rms = np.mean(librosa.feature.rms(y=y))
    
    # 3. Create Feature Array (Must match the order in train_ann.py)
    # Ensure BPM is a float, not a list
    bpm = float(tempo[0]) if isinstance(tempo, (list, np.ndarray)) else float(tempo)
    features = np.array([[bpm, mfcc, centroid, rolloff, chroma, zcr, rms]])
    
    # 4. Scale and Predict
    scaled_features = scaler.transform(features) # Make sure you use the same scaler!
    prediction = model.predict(scaled_features)
    
    return prediction

# --- UI SECTION ---
st.title("🎵 Music Mood Classifier")
st.write("Upload a song from your current Device!")

uploaded_file = st.file_uploader("Choose an MP3 file", type="mp3")

if uploaded_file is not None:
    with open("temp.mp3", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.audio(uploaded_file)
    
    with st.spinner('Analyzing the math of the song...'):
        # Calling the new function
        mood, conf, bpm, text, bright = predict_mood("temp.mp3", model, scaler, encoder)
        
        st.success(f"🔥 Predicted Mood: **{mood.upper()}**")
        st.info(f"🧠 AI Confidence: **{round(conf, 2)}%**")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("BPM", round(float(bpm), 1))
        col2.metric("Texture (MFCC)", round(float(text), 1))
        col3.metric("Brightness", round(float(bright), 1))