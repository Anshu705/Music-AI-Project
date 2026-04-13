import streamlit as st
import librosa
import numpy as np
import keras
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os
import time

# --- 1. PAGE CONFIG & PREMIUM CSS ---
st.set_page_config(page_title="VibeSynth AI | Manan", page_icon="🎵", layout="wide")

st.markdown("""
<style>
    .stApp { background: radial-gradient(circle at 20% 30%, #1e1e2f 0%, #0d0d12 100%); color: white; }
    [data-testid="stVerticalBlock"] > div:has(div.element-container) {
        background: rgba(255, 255, 255, 0.05); backdrop-filter: blur(15px);
        border-radius: 25px; padding: 25px; border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .stButton>button {
        background: linear-gradient(90deg, #00f2ea, #0072ff); border: none;
        color: white; border-radius: 50px; font-weight: bold; width: 100%;
    }
    h1, h2, h3 { color: #00f2ea !important; }
</style>
""", unsafe_allow_html=True)

# --- 2. AI BRAIN INITIALIZATION ---
@st.cache_resource
def load_ai():
    if not os.path.exists('music_mood_model_1000.keras'):
        st.error("Model file missing! Please upload music_mood_model_1000.keras")
        return None, None, None
    
    model = keras.models.load_model('music_mood_model_1000.keras')
    data = pd.read_csv('music_database_1000.csv')
    
    # Clean BPM and fit preprocessing
    data['BPM'] = data['BPM'].apply(lambda x: float(str(x).replace('[', '').replace(']', '')))
    X_raw = data[['BPM', 'MFCC', 'Centroid', 'Rolloff', 'Chroma', 'ZCR', 'RMS']].values
    scaler = StandardScaler().fit(X_raw)
    encoder = LabelEncoder().fit(data['Label'])
    
    return model, scaler, encoder

model, scaler, encoder = load_ai()

# --- 3. CORE PREDICTION LOGIC ---
def predict_mood(file_path):
    y, sr = librosa.load(file_path, duration=30)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    bpm = float(tempo[0]) if isinstance(tempo, (list, np.ndarray)) else float(tempo)
    
    # Feature Extraction (7 Features)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13))
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    rms = np.mean(librosa.feature.rms(y=y))
    
    # Prediction
    features = np.array([[bpm, mfcc, centroid, rolloff, chroma, zcr, rms]])
    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)
    
    mood_label = encoder.inverse_transform([np.argmax(prediction)])[0]
    confidence = np.max(prediction) * 100
    
    return mood_label, confidence, bpm, mfcc, centroid

# --- 4. SIDEBAR (Subscriptions & Profile) ---
with st.sidebar:
    st.title("👤 Manan Bansal")
    st.caption("Arya College | AI & DS")
    st.markdown("---")
    menu = st.radio("Navigation", ["🏠 Home", "🎹 Artist Studio", "💎 Subscriptions", "📊 Data Logs"])
    
    st.markdown("---")
    st.write("### ⏱️ AI Usage")
    st.progress(75, text="45 Mins Remaining")

# --- 5. MAIN DASHBOARD ---
if menu == "🏠 Home":
    st.title("🎵 Global Genre Classifier")
    st.write("Analyze songs against our **1,000-Song** Neural Network.")
    
    uploaded_file = st.file_uploader("Upload an MP3 for Global Analysis", type="mp3")

    if uploaded_file:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.audio(uploaded_file)
            if st.button("🚀 Analyze Frequency"):
                with open("temp.mp3", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                with st.spinner("Decoding audio..."):
                    mood, conf, bpm, mfcc, bright = predict_mood("temp.mp3")
                    st.session_state['last_mood'] = mood
                    
                    st.success(f"Genre: {mood.upper()}")
                    st.metric("Confidence", f"{round(conf, 1)}%")

        with col2:
            if 'last_mood' in st.session_state:
                st.write("### 📊 Audio DNA")
                st.metric("Tempo", f"{round(bpm, 1)} BPM")
                st.metric("Complexity", f"{round(mfcc, 1)}")
                st.metric("Brightness", f"{round(bright, 1)}")

elif menu == "🎹 Artist Studio":
    st.title("🎨 Creator Dashboard")
    st.write("Upload vocals to generate AI music or use virtual instruments.")
    
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("➕ Vocal-to-Composition")
        vocal = st.file_uploader("Upload Vocal Track", type=['mp3', 'wav'])
        if vocal:
            st.info("Analyzing pitch... AI is generating a back-track.")
    with c2:
        st.subheader("🎹 Instrument Studio")
        inst = st.selectbox("Select Gear", ["Grand Piano", "Electronic Synth", "Indian Sitar"])
        if st.button("Open Piano GUI"):
            st.toast("Connecting to MIDI interface...")

elif menu == "💎 Subscriptions":
    st.title("💎 Choose Your Power")
    p1, p2, p3, p4 = st.columns(4)
    p1.markdown("### Listener\n**₹7**\n14 Days\n\n*Ad-free AI*")
    p2.markdown("### UI Pro\n**₹14**\n14 Days\n\n*Custom Themes*")
    p3.markdown("### Artist\n**₹21**\n14 Days\n\n*All Instruments*")
    p4.markdown("### ALL-IN\n**₹45**\n30 Days\n\n*45 Mins AI Gen*")
    
    st.button("Checkout with Razorpay / UPI")

elif