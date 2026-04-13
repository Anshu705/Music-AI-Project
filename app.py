import streamlit as st
import librosa
import numpy as np
import keras
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

st.set_page_config(page_title="Manan's Music AI", page_icon="🎵")

# --- 1. SETUP & LOADING ---
@st.cache_resource
def load_ai():
    # Load the 1,000-song model
    model = keras.models.load_model('music_mood_model_1000.keras')
    
    # Load database to re-initialize the Scaler and Encoder
    data = pd.read_csv('music_database_1000.csv')
    
    # Fix the BPM string-brackets issue
    data['BPM'] = data['BPM'].apply(lambda x: float(str(x).replace('[', '').replace(']', '')))
    
    # Fit Scaler on all 7 features
    X_raw = data[['BPM', 'MFCC', 'Centroid', 'Rolloff', 'Chroma', 'ZCR', 'RMS']].values
    scaler = StandardScaler().fit(X_raw)
    
    # Fit Encoder on the 10 Genres
    encoder = LabelEncoder()
    encoder.fit(data['Label'])
    
    return model, scaler, encoder

# Initialize the "Brain"
model, scaler, encoder = load_ai()

# --- 2. THE PREDICTION LOGIC ---
def predict_mood(file_path):
    # Load 30 seconds of audio
    y, sr = librosa.load(file_path, duration=30)
    
    # Extract 7 Professional Features
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13))
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    rms = np.mean(librosa.feature.rms(y=y))
    
    # Format and Scale
    bpm = float(tempo[0]) if isinstance(tempo, (list, np.ndarray)) else float(tempo)
    features = np.array([[bpm, mfcc, centroid, rolloff, chroma, zcr, rms]])
    scaled_features = scaler.transform(features)
    
    # Run Inference
    prediction = model.predict(scaled_features)
    
    # Get Label and Confidence
    mood_label = encoder.inverse_transform([np.argmax(prediction)])[0]
    confidence = np.max(prediction) * 100
    
    # RETURN 5 VALUES to match the UI Unpacking (Line 62)
    return mood_label, confidence, bpm, mfcc, centroid

# --- 3. UI SECTION ---
st.title("🎵 Music Genre & Mood Classifier")
st.write("Powered by Manan's **1,000-Song** Neural Network")

uploaded_file = st.file_uploader("Choose an MP3 file", type="mp3")

if uploaded_file is not None:
    # Save file locally for processing
    with open("temp.mp3", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.audio(uploaded_file)
    
    if st.button("Start AI Analysis"):
        with st.spinner('Analyzing audio frequencies...'):
            try:
                # Corrected call: only passing the file path as the function expects
                mood, conf, bpm, text, bright = predict_mood("temp.mp3")
                
                st.success(f"🔥 Predicted Genre: **{mood.upper()}**")
                st.info(f"🧠 AI Confidence: **{round(conf, 2)}%**")
                
                # Metric Dashboard
                c1, c2, c3 = st.columns(3)
                c1.metric("BPM", round(float(bpm), 1))
                c2.metric("Texture (MFCC)", round(float(text), 1))
                c3.metric("Brightness", round(float(bright), 1))
            except Exception as e:
                st.error(f"Analysis failed: {e}")