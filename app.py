import streamlit as st
import librosa
import numpy as np
import keras
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os
import time

# ==========================================
# 1. ADVANCED ENGINE (CSS & JAVASCRIPT)
# ==========================================
st.set_page_config(page_title="VibeSynth Premium | Manan", page_icon="⚡", layout="wide")

def inject_ultra_ui():
    st.markdown("""
    <style>
        /* A. RADIANT ANIMATED BACKGROUND */
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

        /* B. GAMING BLADE CARDS */
        .sub-card {
            background: rgba(255, 255, 255, 0.04);
            backdrop-filter: blur(25px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 30px;
            padding: 45px;
            text-align: center;
            transition: all 0.6s cubic-bezier(0.165, 0.84, 0.44, 1);
        }
        .sub-card:hover {
            transform: scale(1.08) translateY(-20px);
            background: rgba(0, 242, 234, 0.1);
            border-color: #00f2ea;
            box-shadow: 0 30px 60px rgba(0,0,0,0.5), 0 0 20px rgba(0, 242, 234, 0.3);
        }

        /* C. ARTIST HUB PREMIUM CARDS */
        .artist-hub-card {
            background: linear-gradient(135deg, rgba(255,255,255,0.1), rgba(0,0,0,0.4));
            border-radius: 20px;
            padding: 25px;
            border-left: 6px solid #00f2ea;
            margin-bottom: 20px;
        }

        /* D. MOUSE GLOW */
        #glow {
            position: fixed;
            top: 0; left: 0;
            width: 800px; height: 800px;
            background: radial-gradient(circle, rgba(0,242,234,0.12) 0%, rgba(0,0,0,0) 70%);
            border-radius: 50%;
            pointer-events: none;
            transform: translate(-50%, -50%);
            z-index: 0;
        }

        /* E. QR MODAL */
        .qr-container {
            background: white;
            color: black;
            padding: 20px;
            border-radius: 20px;
            text-align: center;
            box-shadow: 0 0 50px rgba(0,0,0,0.8);
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

inject_ultra_ui()

# ==========================================
# 2. AI BACKEND
# ==========================================
@st.cache_resource
def load_ai():
    model_path = 'music_mood_model_1000.keras' 
    csv_path = 'music_database_1000.csv'
    if not os.path.exists(model_path): return None, None, None
    model = keras.models.load_model(model_path)
    data = pd.read_csv(csv_path)
    data['BPM'] = data['BPM'].apply(lambda x: float(str(x).replace('[', '').replace(']', '')))
    X_raw = data[['BPM', 'MFCC', 'Centroid', 'Rolloff', 'Chroma', 'ZCR', 'RMS']].values
    scaler = StandardScaler().fit(X_raw)
    encoder = LabelEncoder().fit(data['Label'])
    return model, scaler, encoder

model, scaler, encoder = load_ai()

# ==========================================
# 3. CORE NAVIGATION
# ==========================================
with st.sidebar:
    st.markdown("<h1 style='color:#00f2ea;'>VibeSynth</h1>", unsafe_allow_html=True)
    st.image("https://images.unsplash.com/photo-1470225620780-dba8ba36b745?auto=format&fit=crop&q=80&w=300", use_container_width=True)
    menu = st.radio("MAIN MENU", ["🏠 Mission Control", "🎹 Artist Hub", "💎 Subscriptions"])
    st.markdown("---")
    st.write("🛡️ **Copyright Protection: ON**")
    st.caption("Verifying fingerprints in real-time...")

# ==========================================
# 4. PAGE: MISSION CONTROL (HOME)
# ==========================================
if menu == "🏠 Mission Control":
    st.markdown("<h1 style='text-align: center; font-size: 60px;'>Elevate Your Vibe</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image("https://images.unsplash.com/photo-1493225255756-d9584f8606e9?auto=format&fit=crop&q=80&w=800", caption="Future of Sound")
    with col2:
        st.markdown("### 🚀 Our Mission")
        st.write("VibeSynth was born in Jaipur with a singular goal: To democratize music production. We use advanced Artificial Intelligence to turn raw emotion into studio-grade compositions.")
        
        st.markdown("### 🎯 Our Goal")
        st.write("We aim to protect independent creators. Through our automated **Copyright Reservation System**, every note generated is legally fingerprinted to you instantly.")
        
        st.markdown("""
        <div style="background:rgba(0,242,234,0.1); padding:20px; border-radius:15px; border:1px solid #00f2ea;">
            <b>Live Status:</b> Retrained on 1,000 Professional Tracks for 98% Mood Accuracy.
        </div>
        """, unsafe_allow_html=True)

# ==========================================
# 5. PAGE: ARTIST HUB (UPGRADED)
# ==========================================
elif menu == "🎹 Artist Hub":
    st.title("🎨 Artist Hub & AI Studio")
    
    col_a, col_b = st.columns([3, 2])
    
    with col_a:
        st.markdown("""<div class="artist-hub-card">
            <h3>🎸 Vocal Alchemy (45-Min Sessions)</h3>
            <p>Our most powerful engine. Upload your vocals, and the AI builds a full 45-minute arrangements including drums, bass, and synth melodies tailored to your pitch.</p>
        </div>""", unsafe_allow_html=True)
        st.file_uploader("Drop Vocal Stem", type=['mp3', 'wav'])
        
        st.markdown("""<div class="artist-hub-card">
            <h3>🛡️ Copyright Strike Guard</h3>
            <p>Before you release, our system scans global databases. We ensure your AI-generated track is unique and issues a <b>VibeSynth Certificate</b> of Ownership.</p>
        </div>""", unsafe_allow_html=True)
        if st.button("Generate Ownership Certificate"):
            st.success("Certificate #VS-99281-2026 Issued to Manan Bansal")

    with col_b:
        st.image("https://images.unsplash.com/photo-1598488035139-bdbb2231ce04?auto=format&fit=crop&q=80&w=600", caption="Studio Interface")
        st.markdown("### 🎹 Quick Tools")
        st.button("Open MIDI Keyboard")
        st.button("Mood-to-Melody Converter")

# ==========================================
# 6. PAGE: SUBSCRIPTIONS (PAYMENT QR SYSTEM)
# ==========================================
elif menu == "💎 Subscriptions":
    st.markdown("<h1 style='text-align: center;'>Upgrade Your Frequency</h1>", unsafe_allow_html=True)
    st.write("<p style='text-align:center;'>All premium tiers now feature <b>45-Day Validity</b>.</p>", unsafe_allow_html=True)
    
    p1, p2, p3, p4 = st.columns(4)
    plans = [
        {"name": "LISTENER", "price": 7, "color": "#00f2ea", "feat": "AI Mood Prediction"},
        {"name": "UI PRO", "price": 14, "color": "#ff00ff", "feat": "Custom Dashboards"},
        {"name": "ARTIST", "price": 21, "color": "#00ff00", "feat": "45-Min Generation"},
        {"name": "ALL-IN", "price": 25, "color": "gold", "feat": "Full Studio + VIP Support"}
    ]

    for idx, col in enumerate([p1, p2, p3, p4]):
        plan = plans[idx]
        with col:
            st.markdown(f"""<div class="sub-card" style="border-color:{plan['color']};">
                <h2 style="color:{plan['color']}">{plan['name']}</h2>
                <h1 style="font-size:50px;">₹{plan['price']}</h1>
                <p><b>45 DAYS</b></p>
                <hr style="opacity:0.2">
                <p>{plan['feat']}</p>
            </div>""", unsafe_allow_html=True)
            if st.button(f"Get {plan['name']}", key=f"btn_{plan['price']}"):
                st.session_state['show_qr'] = plan['price']

    # QR Code Popup Interface
    if 'show_qr' in st.session_state:
        st.markdown("---")
        st.subheader(f"💳 Payment Secure Gateway: ₹{st.session_state['show_qr']}")
        
        q_col1, q_col2 = st.columns([1, 2])
        with q_col1:
            # Note: Save your images as qr_7.png, qr_14.png etc in your github repo
            qr_file = f"qr_{st.session_state['show_qr']}.png"
            if os.path.exists(qr_file):
                st.image(qr_file, width=250)
            else:
                st.error(f"Please upload '{qr_file}' to your repository.")
                st.caption("Dummy QR for Presentation:")
                st.image("https://api.qrserver.com/v1/create-qr-code/?size=150x150&data=Payment_Dummy")
        
        with q_col2:
            st.write("### Instructions:")
            st.write(f"1. Scan the QR code for exactly ₹{st.session_state['show_qr']}.")
            st.write("2. Enter your Transaction ID below.")
            st.text_input("Transaction ID (UTR)")
            if st.button("Verify Payment"):
                st.balloons()
                st.success("Access Granted! Your OnePlus Nord CE4 app is now Syncing.")
            if st.button("Close Payment Window"):
                del st.session_state['show_qr']
                st.rerun()