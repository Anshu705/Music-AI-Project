import streamlit as st
import librosa
import numpy as np
import keras
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os
import time

# ==========================================
# 1. SUPER UI ENGINE (CSS & JAVASCRIPT)
# ==========================================
st.set_page_config(page_title="VibeSynth Ultra | Manan", page_icon="⚡", layout="wide")

def inject_super_design():
    st.markdown("""
    <style>
        /* A. DYNAMIC GRADIENT BACKGROUND */
        .stApp {
            background: linear-gradient(-45deg, #0f0c29, #302b63, #24243e, #00d2ff);
            background-size: 400% 400%;
            animation: gradientBG 15s ease infinite;
            color: white;
            font-family: 'Inter', sans-serif;
        }
        @keyframes gradientBG {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        /* B. CLOUD GAMING 'BLADE' SUBSCRIPTION CARDS */
        .sub-card {
            background: rgba(255, 255, 255, 0.03);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 24px;
            padding: 40px;
            text-align: center;
            transition: all 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            height: 100%;
        }
        .sub-card:hover {
            transform: scale(1.08) translateY(-15px);
            background: rgba(255, 255, 255, 0.12);
            border-color: #00d2ff;
            box-shadow: 0 20px 50px rgba(0, 210, 255, 0.3);
        }

        /* C. SUNO-STYLE MUSIC ROW */
        .music-row {
            display: flex;
            align-items: center;
            background: rgba(255, 255, 255, 0.05);
            padding: 15px 25px;
            border-radius: 15px;
            margin-bottom: 12px;
            border: 1px solid transparent;
            transition: 0.3s;
        }
        .music-row:hover {
            background: rgba(0, 210, 255, 0.1);
            border-color: #00d2ff;
            cursor: pointer;
        }

        /* D. LOGIN GLASS CARD */
        .login-box {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(30px);
            border-radius: 30px;
            padding: 60px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            max-width: 500px;
            margin: 100px auto;
            text-align: center;
        }

        /* E. MOUSE TRACKING GLOW */
        #glow {
            position: fixed;
            top: 0; left: 0;
            width: 700px; height: 700px;
            background: radial-gradient(circle, rgba(0,210,255,0.15) 0%, rgba(0,0,0,0) 70%);
            border-radius: 50%;
            pointer-events: none;
            transform: translate(-50%, -50%);
            z-index: 0;
        }
        
        /* Sidebar Polish */
        [data-testid="stSidebar"] {
            background: rgba(0, 0, 0, 0.6) !important;
            backdrop-filter: blur(10px);
        }
    </style>
    
    <div id="glow"></div>
    <script>
        const glow = document.getElementById('glow');
        document.addEventListener('mousemove', (e) => {
            glow.style.left = e.clientX + 'px';
            glow.style.top = e.clientY + 'px';
        });
    </script>
    """, unsafe_allow_html=True)

inject_super_design()

# ==========================================
# 2. AI BACKEND LOGIC (7-FEATURE ENGINE)
# ==========================================
@st.cache_resource
def load_ai():
    # Use 1000-song database for the "Super" version
    model_path = 'music_mood_model_1000.keras' 
    csv_path = 'music_database_1000.csv'
    
    if not os.path.exists(model_path):
        st.error("⚠️ AI Brain (Keras model) not found!")
        return None, None, None

    model = keras.models.load_model(model_path)
    data = pd.read_csv(csv_path)
    
    # Preprocessing
    data['BPM'] = data['BPM'].apply(lambda x: float(str(x).replace('[', '').replace(']', '')))
    X_raw = data[['BPM', 'MFCC', 'Centroid', 'Rolloff', 'Chroma', 'ZCR', 'RMS']].values
    scaler = StandardScaler().fit(X_raw)
    encoder = LabelEncoder().fit(data['Label'])
    
    return model, scaler, encoder

model, scaler, encoder = load_ai()

def predict_mood(file_path):
    y, sr = librosa.load(file_path, duration=30)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    bpm = float(tempo[0]) if isinstance(tempo, (list, np.ndarray)) else float(tempo)
    
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

# ==========================================
# 3. APP NAVIGATION & AUTH
# ==========================================
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if not st.session_state['logged_in']:
    st.markdown('<div class="login-box">', unsafe_allow_html=True)
    st.markdown("<h1>⚡ VibeSynth Ultra</h1>", unsafe_allow_html=True)
    st.write("Next-Gen AI Music Prediction")
    user = st.text_input("Username")
    pwd = st.text_input("Password", type="password")
    if st.button("UNLOCK ACCESS"):
        st.session_state['logged_in'] = True
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

else:
    # Top Navigation Row (YT Music Inspired)
    t1, t2, t3 = st.columns([3, 1, 1])
    with t1: st.markdown("<h2 style='color:#00d2ff;'>Explore the Vibe</h2>", unsafe_allow_html=True)
    with t2: st.button("🏠 Home", use_container_width=True)
    with t3: 
        if st.button("🚪 Logout"):
            st.session_state['logged_in'] = False
            st.rerun()

    # --- SIDEBAR ---
    with st.sidebar:
        st.title("👤 Manan Bansal")
        st.caption("AI & DS Engineer")
        st.markdown("---")
        menu = st.radio("Navigation", ["🎧 Home", "🎹 Artist Studio", "💎 Subscriptions", "📊 Data Logs"])
        st.markdown("---")
        st.write("### ⏱️ Usage Tracker")
        st.progress(75, text="45 Mins Premium AI Time Remaining")

    # ==========================================
    # 4. MAIN DASHBOARD CONTENT
    # ==========================================
    if menu == "🎧 Home":
        st.markdown("### 🚀 Global Genre Analysis")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_file = st.file_uploader("Drop your track for AI Decomposition", type="mp3")
            if uploaded_file:
                st.audio(uploaded_file)
                if st.button("🔥 RUN SUPER ANALYSIS"):
                    with open("temp.mp3", "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    with st.spinner('Decrypting Audio Frequencies...'):
                        mood, conf, bpm, mfcc, bright = predict_mood("temp.mp3")
                        st.session_state['res'] = (mood, conf, bpm, mfcc, bright)

        with col2:
            if 'res' in st.session_state:
                m, c, b, mf, br = st.session_state['res']
                st.markdown(f"### Results for {m.upper()}")
                st.metric("AI Confidence", f"{round(c, 1)}%")
                st.metric("Tempo", f"{round(b, 1)} BPM")
                st.metric("Brightness", f"{round(br, 1)}")

        st.markdown("### 🎵 Recently Processed (Suno Style)")
        for i in range(3):
            st.markdown(f"""
            <div class="music-row">
                <div style="flex: 0; margin-right: 20px;"><img src="https://via.placeholder.com/50/00d2ff/ffffff?text=♫" width="50" style="border-radius:10px;"></div>
                <div style="flex: 2;"><b>Global Track 00{i+1}</b><br><small>Detected by VibeSynth Neural Net</small></div>
                <div style="flex: 1; color: #00d2ff;">Genre: <b>Active</b></div>
                <div style="flex: 0;">03:45</div>
            </div>
            """, unsafe_allow_html=True)

    elif menu == "🎹 Artist Studio":
        st.title("🎨 Artist Creation Hub")
        st.write("Tools for professional music production.")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("""<div class="sub-card" style="text-align:left;">
                <h4>➕ Vocal-to-Composition</h4>
                <p>Upload clean vocals and let the AI generate 45 minutes of backing tracks.</p>
            </div>""", unsafe_allow_html=True)
            st.file_uploader("Upload Vocals", type=['mp3', 'wav'])
        with c2:
            st.markdown("""<div class="sub-card" style="text-align:left;">
                <h4>🎹 Virtual Instruments</h4>
                <p>Snapdragon-optimized Digital Audio Workstation tools.</p>
            </div>""", unsafe_allow_html=True)
            if st.button("Open Piano GUI"): st.toast("Connecting MIDI Interface...")

    elif menu == "💎 Subscriptions":
        st.title("💎 Choose Your Power Level")
        p1, p2, p3, p4 = st.columns(4)
        
        plans = [
            ("₹7", "LISTENER", "#00d2ff", "AI Mood Engine"),
            ("₹14", "UI PRO", "#ff00ff", "Glass Themes"),
            ("₹21", "ARTIST", "#00ff00", "Studio Access"),
            ("₹45", "ALL-IN", "gold", "Everything + 45min")
        ]
        
        for idx, col in enumerate([p1, p2, p3, p4]):
            price, name, color, perk = plans[idx]
            with col:
                st.markdown(f"""<div class="sub-card" style="border-color:{color};">
                    <h3 style="color:{color}">{name}</h3>
                    <h1>{price}</h1>
                    <p>14 DAYS</p>
                    <hr style="opacity:0.1">
                    <p>{perk}</p>
                </div>""", unsafe_allow_html=True)
                st.button(f"Upgrade to {name}", key=f"sub_{idx}")

    elif menu == "📊 Data Logs":
        st.title("📈 Training & Feedback Logs")
        if os.path.exists('music_database_1000.csv'):
            df = pd.read_csv('music_database_1000.csv')
            st.dataframe(df.tail(20), use_container_width=True)
        else:
            st.write("Initializing Database...")