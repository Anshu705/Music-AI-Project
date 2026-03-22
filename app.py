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
def predict_mood(file_path, model, scaler, encoder):
    y, sr = librosa.load(file_path, duration=30)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    
    val_tempo = tempo[0] if isinstance(tempo, np.ndarray) else tempo
    features = scaler.transform([[val_tempo, np.mean(mfcc), np.mean(centroid)]])
    
    prediction = model.predict(features, verbose=0)[0]
    index = np.argmax(prediction)
    mood = encoder.inverse_transform([index])[0]
    confidence = prediction[index] * 100
    
    return mood, confidence, val_tempo, np.mean(mfcc), np.mean(centroid)

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