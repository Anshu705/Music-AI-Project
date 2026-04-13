import streamlit as st
import librosa
import numpy as np
import keras
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

# --- 1. PAGE CONFIG & DYNAMIC DESIGN ---
st.set_page_config(page_title="VibeSynth AI | Manan", page_icon="🎵", layout="wide")

# Get UI preference from session state (Default to standard Glassmorphism)
if 'ui_color' not in st.session_state:
    st.session_state['ui_color'] = "#00f2ea"

st.markdown(f"""
<style>
    .stApp {{ background: radial-gradient(circle at 20% 30%, #1e1e2f 0%, #0d0d12 100%); color: white; }}
    [data-testid="stVerticalBlock"] > div:has(div.element-container) {{
        background: rgba(255, 255, 255, 0.05); backdrop-filter: blur(15px);
        border-radius: 25px; padding: 25px; border: 1px solid rgba(255, 255, 255, 0.1);
    }}
    .stButton>button {{
        background: linear-gradient(90deg, {st.session_state['ui_color']}, #0072ff); border: none;
        color: white; border-radius: 50px; font-weight: bold; width: 100%;
    }}
    h1, h2, h3 {{ color: {st.session_state['ui_color']} !important; }}
</style>
""", unsafe_allow_html=True)

# --- 2. AI BRAIN LOADING (FIXED SCALER) ---
@st.cache_resource
def load_ai():
    # Updating to your 1000-song dataset filenames
    model_path = 'music_mood_model_1000.keras' 
    csv_path = 'music_database_1000.csv'
    
    if not os.path.exists(model_path) or not os.path.exists(csv_path):
        st.error("Model or Database files not found on GitHub!")
        return None, None, None

    model = keras.models.load_model(model_path)
    data = pd.read_csv(csv_path)
    
    # Preprocessing: Remove brackets from BPM
    data['BPM'] = data['BPM'].apply(lambda x: float(str(x).replace('[', '').replace(']', '')))
    
    # 🛠️ FIX: Defining the Scaler properly
    X_raw = data[['BPM', 'MFCC', 'Centroid', 'Rolloff', 'Chroma', 'ZCR', 'RMS']].values
    scaler = StandardScaler().fit(X_raw) 
    
    encoder = LabelEncoder().fit(data['Label'])
    
    return model, scaler, encoder

model, scaler, encoder = load_ai()

# --- 3. PREDICTION FUNCTION ---
def predict_mood(file_path):
    y, sr = librosa.load(file_path, duration=30)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    bpm = float(tempo[0]) if isinstance(tempo, (list, np.ndarray)) else float(tempo)
    
    # Extracting all 7 Features
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13))
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    rms = np.mean(librosa.feature.rms(y=y))
    
    features = np.array([[bpm, mfcc, centroid, rolloff, chroma, zcr, rms]])
    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)
    
    label = encoder.inverse_transform([np.argmax(prediction)])[0]
    conf = np.max(prediction) * 100
    return label, conf, bpm, mfcc, centroid

# --- 4. SIDEBAR NAVIGATION ---
with st.sidebar:
    st.title("👤 Manan Bansal")
    st.caption("B.Tech AI & DS | ACEIT")
    st.markdown("---")
    menu = st.radio("Go To", ["🏠 Home", "🎹 Artist Studio", "💎 Subscriptions", "📊 Data Logs"])
    st.markdown("---")
    
    # Logic for UI Pro Users (Subscription Feature)
    if menu == "💎 Subscriptions":
         st.write("### 🎨 UI Customizer (PRO)")
         new_color = st.color_picker("Pick your Brand Color", st.session_state['ui_color'])
         if st.button("Apply Theme"):
             st.session_state['ui_color'] = new_color
             st.rerun()

    st.write("### ⏱️ AI Usage")
    st.progress(75, text="45 Mins Remaining")

# --- 5. MAIN LOGIC ---
if menu == "🏠 Home":
    st.title("🎵 Global Genre Classifier")
    st.write("Retrained on the **1,000-Song GTZAN Dataset**.")
    uploaded_file = st.file_uploader("Upload an MP3", type="mp3")

    if uploaded_file:
        col1, col2 = st.columns(2)
        with col1:
            st.audio(uploaded_file)
            if st.button("🚀 Run AI Analysis"):
                with open("temp.mp3", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                mood, conf, bpm, mfcc, bright = predict_mood("temp.mp3")
                st.session_state['result'] = (mood, conf, bpm, mfcc, bright)

        with col2:
            if 'result' in st.session_state:
                m, c, b, mf, br = st.session_state['result']
                st.success(f"Predicted Genre: **{m.upper()}**")
                st.metric("AI Confidence", f"{round(c, 1)}%")
                st.metric("Tempo (Math)", f"{round(b, 1)} BPM")

elif menu == "🎹 Artist Studio":
    st.title("🎨 Creator Studio")
    st.write("Artist tools for recording and AI composition.")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("➕ Vocal to Music")
        vocal = st.file_uploader("Upload Vocals", type=['mp3', 'wav'])
        if vocal: st.info("AI is composing background music...")
    with c2:
        st.subheader("🎹 Virtual Instruments")
        if st.button("Open Piano GUI"): st.toast("Connecting MIDI for OnePlus Nord CE4...")

elif menu == "💎 Subscriptions":
    st.title("💎 Membership Plans")
    p1, p2, p3, p4 = st.columns(4)
    p1.info("₹7\nListener\n(14 Days)")
    p2.warning("₹14\nUI Pro\n(14 Days)")
    p3.success("₹21\nArtist\n(14 Days)")
    p4.error("₹45\nAll-In\n(30 Days)")
    st.button("Checkout with Razorpay / UPI")

elif menu == "📊 Data Logs":
    st.title("📈 Training & Feedback Logs")
    if os.path.exists('music_database_1000.csv'):
        df = pd.read_csv('music_database_1000.csv')
        st.dataframe(df.tail(15), use_container_width=True)
    else:
        st.write("Database not initialized.")