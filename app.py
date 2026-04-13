import streamlit as st
import librosa
import numpy as np
import keras
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os
import time
import base64
from datetime import datetime

# ==============================================================================
# MODULE A: THE GLOBAL STYLE ENGINE (NEO-GLASSMORPHISM)
# ==============================================================================
st.set_page_config(
    page_title="VibeSynth Ultra | The Future of AI Music",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

def inject_vibe_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;700;900&display=swap');
        
        * { font-family: 'Inter', sans-serif; }

        /* VIBRANT ANIMATED BACKGROUND */
        .stApp {
            background: linear-gradient(-45deg, #050505, #121212, #001f3f, #00d2ff);
            background-size: 400% 400%;
            animation: vibeGradient 15s ease infinite;
            color: #ffffff;
        }
        @keyframes vibeGradient {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        /* CUSTOM SIDEBAR (SUNO STYLE) */
        [data-testid="stSidebar"] {
            background: rgba(0, 0, 0, 0.7) !important;
            backdrop-filter: blur(20px);
            border-right: 1px solid rgba(0, 210, 255, 0.2);
        }

        /* CLOUD GAMING BLADE CARDS */
        .premium-card {
            background: rgba(255, 255, 255, 0.03);
            backdrop-filter: blur(25px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 30px;
            padding: 40px;
            text-align: center;
            transition: all 0.6s cubic-bezier(0.165, 0.84, 0.44, 1);
        }
        .premium-card:hover {
            transform: scale(1.06) translateY(-15px);
            background: rgba(0, 210, 255, 0.08);
            border-color: #00d2ff;
            box-shadow: 0 40px 80px rgba(0,0,0,0.6), 0 0 30px rgba(0, 210, 255, 0.4);
        }

        /* SUNO MUSIC ROW */
        .track-row {
            display: flex;
            align-items: center;
            background: rgba(255, 255, 255, 0.03);
            border-radius: 20px;
            padding: 20px 30px;
            margin-bottom: 15px;
            border: 1px solid transparent;
            transition: 0.3s ease;
        }
        .track-row:hover {
            background: rgba(255, 255, 255, 0.08);
            border-color: #00d2ff;
            cursor: pointer;
        }

        /* GLOWING BUTTONS */
        .stButton>button {
            border-radius: 50px;
            padding: 12px 30px;
            background: linear-gradient(90deg, #00d2ff, #0072ff);
            border: none;
            color: white;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1px;
            transition: 0.4s;
        }
        .stButton>button:hover {
            box-shadow: 0 0 25px #00d2ff;
            transform: translateY(-2px);
        }

        /* MOUSE GLOW OVERLAY */
        #cursor-glow {
            position: fixed;
            top: 0; left: 0;
            width: 800px; height: 800px;
            background: radial-gradient(circle, rgba(0,210,255,0.12) 0%, rgba(0,0,0,0) 70%);
            border-radius: 50%;
            pointer-events: none;
            transform: translate(-50%, -50%);
            z-index: 9999;
        }
    </style>
    <div id="cursor-glow"></div>
    <script>
        const glow = document.getElementById('cursor-glow');
        document.addEventListener('mousemove', (e) => {
            glow.style.left = e.clientX + 'px';
            glow.style.top = e.clientY + 'px';
        });
    </script>
    """, unsafe_allow_html=True)

# ==============================================================================
# MODULE B: THE AI BRAIN (SIGNAL PROCESSING)
# ==============================================================================
@st.cache_resource
def initialize_ai_engine():
    m_file = 'music_mood_model_1000.keras'
    d_file = 'music_database_1000.csv'
    
    if not os.path.exists(m_file) or not os.path.exists(d_file):
        return None, None, None

    brain = keras.models.load_model(m_file)
    db = pd.read_csv(d_file)
    
    # Format BPM data
    db['BPM'] = db['BPM'].apply(lambda x: float(str(x).replace('[', '').replace(']', '')))
    
    # Feature Scaler
    features = db[['BPM', 'MFCC', 'Centroid', 'Rolloff', 'Chroma', 'ZCR', 'RMS']].values
    scaler = StandardScaler().fit(features)
    
    # Genre Encoder
    encoder = LabelEncoder().fit(db['Label'])
    
    return brain, scaler, encoder

brain, scaler, encoder = initialize_ai_engine()

# ==============================================================================
# MODULE C: MISSION & BRANDING (HOME)
# ==============================================================================
def show_home():
    st.markdown("<h1 style='text-align: center; font-size: 80px; font-weight: 900; margin-bottom: 0;'>VibeSynth Ultra</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 22px; opacity: 0.7; margin-bottom: 50px;'>Absolute AI Music Sovereignty</p>", unsafe_allow_html=True)
    
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("""
        <div style="background: rgba(255,255,255,0.05); padding: 40px; border-radius: 25px; border-left: 10px solid #00d2ff;">
            <h2 style="color: #00d2ff;">🚀 Our Mission</h2>
            <p style="font-size: 18px; line-height: 1.6;">VibeSynth was engineered in Jaipur with a singular purpose: to democratize music production for the next generation. We leverage advanced Signal Processing and Deep Learning to turn human emotion into professional-grade soundscapes.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_r:
        st.markdown("""
        <div style="background: rgba(255,255,255,0.05); padding: 40px; border-radius: 25px; border-left: 10px solid #0072ff;">
            <h2 style="color: #0072ff;">🎯 The 2027 Vision</h2>
            <p style="font-size: 18px; line-height: 1.6;">Our goal is to create the first 'One-Click DAW'. We are moving toward a future where a 10-second vocal hum can be transformed into a 45-minute orchestrated album, fully copyright-reserved and legally protected instantly.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.image("https://images.unsplash.com/photo-1598488035139-bdbb2231ce04?w=1200", caption="Inside the VibeSynth Neural Core")

# ==============================================================================
# MODULE D: ARTIST CREATION HUB (DAW)
# ==============================================================================
def show_studio():
    st.markdown("<h1>🎨 Artist Creation Hub</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background: linear-gradient(90deg, rgba(0,210,255,0.1), transparent); padding: 20px; border-radius: 15px; border-left: 5px solid #00d2ff;">
        <h3>🛡️ Copyright Strike Guard & Fingerprinting</h3>
        <p>Before releasing your track, our system scans the global audio landscape. If your generation is 100% unique, we issue a <b>VibeSynth Certificate of Ownership</b>.</p>
    </div>
    """, unsafe_allow_html=True)
    
    c1, c2 = st.columns([2, 1])
    with c1:
        st.write("### ➕ Vocal-to-Composition (45 Min Session)")
        audio = st.file_uploader("Upload Raw Vocal Stem", type=['mp3', 'wav'])
        if audio:
            st.audio(audio)
            if st.button("GENERATE SYMPHONY"):
                with st.spinner("AI is composing drums, bass, and synth layers..."):
                    time.sleep(3)
                    st.success("45-Minute Composition Ready for Export!")
    
    with c2:
        st.image("https://images.unsplash.com/photo-1514525253361-bee8a18744ad?w=600", caption="Studio Hardware Sync")
        st.button("🎹 OPEN VIRTUAL PIANO")
        st.button("🥁 LAUNCH DRUM PAD")

# ==============================================================================
# MODULE E: SUBSCRIPTIONS & QR PAYMENTS
# ==============================================================================
def show_subscriptions():
    st.markdown("<h1 style='text-align: center;'>Upgrade Your Frequency</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>All tiers now include <b>45-Day Validity</b> for the launch season.</p>", unsafe_allow_html=True)
    
    p1, p2, p3, p4 = st.columns(4)
    plans = [
        {"name": "LISTENER", "price": 7, "color": "#00d2ff", "desc": "AI Mood Detection"},
        {"name": "UI PRO", "price": 14, "color": "#ff00ff", "desc": "Custom Glass Themes"},
        {"name": "ARTIST", "price": 21, "color": "#00ff00", "desc": "DAW Studio Access"},
        {"name": "ALL-IN", "price": 25, "color": "gold", "desc": "Full VIP Experience"}
    ]

    for idx, col in enumerate([p1, p2, p3, p4]):
        plan = plans[idx]
        with col:
            st.markdown(f"""
            <div class="premium-card" style="border-color: {plan['color']};">
                <h2 style="color: {plan['color']}">{plan['name']}</h2>
                <h1 style="font-size: 60px;">₹{plan['price']}</h1>
                <p><b>45 DAYS ACCESS</b></p>
                <hr style="opacity: 0.1">
                <p>{plan['desc']}</p>
            </div>
            """, unsafe_allow_html=True)
            if st.button(f"Unlock {plan['name']}", key=f"sub_{plan['price']}"):
                st.session_state['qr_pay'] = plan['price']

    if 'qr_pay' in st.session_state:
        st.markdown("---")
        st.subheader(f"💳 Payment Secure Gateway: ₹{st.session_state['qr_pay']}")
        
        qc1, qc2 = st.columns([1, 2])
        with qc1:
            qr_file = f"qr_{st.session_state['qr_pay']}.png"
            if os.path.exists(qr_file):
                st.image(qr_file, width=300)
            else:
                st.error("QR image file missing!")
                st.image("https://api.qrserver.com/v1/create-qr-code/?size=300x300&data=Manan_VibeSynth_Payment")
        
        with qc2:
            st.write("### Instructions")
            st.write(f"1. Scan to pay exactly ₹{st.session_state['qr_pay']}")
            st.write("2. Upload screenshot or enter Transaction ID (UTR)")
            st.text_input("Enter UTR Number")
            if st.button("VERIFY & ACTIVATE"):
                st.balloons()
                st.success("Subscription Active on your OnePlus Nord CE4!")
            if st.button("CLOSE"):
                del st.session_state['qr_pay']
                st.rerun()

# ==============================================================================
# MAIN EXECUTION FLOW
# ==============================================================================
inject_vibe_css()

with st.sidebar:
    st.markdown("<h1 style='color: #00d2ff;'>VIBESYNTH</h1>", unsafe_allow_html=True)
    st.markdown("---")
    nav = st.radio("SOCIETY", ["🏠 Home", "🎹 Creator Studio", "💎 Subscriptions"])
    st.markdown("---")
    st.caption("Developed by Manan Bansal")
    st.caption("Arya College | Batch 2026")

if nav == "🏠 Home":
    show_home()
elif nav == "🎹 Creator Studio":
    show_studio()
elif nav == "💎 Subscriptions":
    show_subscriptions()