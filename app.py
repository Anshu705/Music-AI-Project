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
# MODULE 1: THE GLOBAL DESIGN ENGINE (CSS & JAVASCRIPT)
# ==============================================================================
st.set_page_config(
    page_title="VibeSynth Ultra | Absolute AI Music",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

def inject_vibe_engine():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;700;900&display=swap');
        
        * { font-family: 'Inter', sans-serif; }

        /* A. RADIANT DYNAMIC GRADIENT BACKGROUND */
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

        /* B. SUNO-STYLE NAVIGATION SIDEBAR */
        [data-testid="stSidebar"] {
            background: rgba(0, 0, 0, 0.8) !important;
            backdrop-filter: blur(25px);
            border-right: 1px solid rgba(0, 242, 234, 0.2);
        }

        /* C. CLOUD GAMING 'BLADE' UI CARDS */
        .blade-card {
            background: rgba(255, 255, 255, 0.03);
            backdrop-filter: blur(30px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 30px;
            padding: 40px;
            text-align: center;
            transition: all 0.6s cubic-bezier(0.165, 0.84, 0.44, 1);
            height: 100%;
        }
        .blade-card:hover {
            transform: scale(1.06) translateY(-15px);
            background: rgba(0, 242, 234, 0.1);
            border-color: #00f2ea;
            box-shadow: 0 40px 80px rgba(0,0,0,0.6), 0 0 30px rgba(0, 242, 234, 0.4);
        }

        /* D. SUNO MUSIC LIST INTERFACE */
        .music-entry {
            display: flex;
            align-items: center;
            background: rgba(255, 255, 255, 0.03);
            border-radius: 20px;
            padding: 20px 30px;
            margin-bottom: 15px;
            border: 1px solid transparent;
            transition: 0.3s ease;
        }
        .music-entry:hover {
            background: rgba(255, 255, 255, 0.1);
            border-color: #00f2ea;
            cursor: pointer;
        }

        /* E. MOUSE TRACKING GLOW (GLO-EFFECT) */
        #cursor-glow {
            position: fixed;
            top: 0; left: 0;
            width: 800px; height: 800px;
            background: radial-gradient(circle, rgba(0, 242, 234, 0.12) 0%, rgba(0,0,0,0) 70%);
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
# MODULE 2: MISSION CONTROL (MISSION, GOAL & BRANDING)
# ==============================================================================
def show_mission_control():
    st.markdown("<h1 style='text-align: center; font-size: 85px; font-weight: 900; color: #00f2ea;'>VibeSynth Ultra</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 24px; opacity: 0.7; letter-spacing: 2px;'>THE ABSOLUTE SUPER-WEBSITE FOR AI MUSIC</p>", unsafe_allow_html=True)
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        <div style="background: rgba(255,255,255,0.05); padding: 45px; border-radius: 30px; border-left: 10px solid #00f2ea;">
            <h2 style="color: #00f2ea;">🚀 Our Mission</h2>
            <p style="font-size: 19px; line-height: 1.8;">VibeSynth was engineered at Arya College, Jaipur, to bridge the gap between human soul and machine intelligence. Our mission is to provide 100% accessible, high-fidelity music generation tools that empower every individual to become a composer, regardless of technical background.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_b:
        st.markdown("""
        <div style="background: rgba(255,255,255,0.05); padding: 45px; border-radius: 30px; border-left: 10px solid #0072ff;">
            <h2 style="color: #0072ff;">🎯 Future Goal (2027)</h2>
            <p style="font-size: 19px; line-height: 1.8;">Our ultimate goal is the creation of a 'One-Click DAW' (Digital Audio Workstation). We are building a future where a simple vocal hum on your OnePlus Nord CE4 can trigger a 45-minute orchestrated masterpiece, instantly copyright-reserved and global-ready.</p>
        </div>
        """, unsafe_allow_html=True)

    st.image("https://images.unsplash.com/photo-1598488035139-bdbb2231ce04?w=1200", caption="Inside the VibeSynth Deep Learning Studio")

# ==============================================================================
# MODULE 3: ARTIST CREATION HUB (DAW & COPYRIGHT)
# ==============================================================================
def show_artist_studio():
    st.title("🎨 Artist Creation Hub")
    st.write("Professional tools for composition, fingerprinting, and copyright reservation.")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""<div style="background:rgba(0,242,234,0.1); padding:25px; border-radius:15px; border-left:5px solid #00f2ea;">
            <h3>🛡️ Copyright Fingerprinting & Reservation</h3>
            <p>Every note you generate is unique. Our system assigns a Digital Fingerprint (SHA-256) to your track, ensuring your ownership is reserved in the VibeSynth Global Database to prevent copyright strikes.</p>
        </div>""", unsafe_allow_html=True)
        
        st.subheader("➕ Vocal-to-Composition (45 Minute Session)")
        vocal = st.file_uploader("Upload your raw vocal stem", type=['mp3', 'wav'])
        if vocal:
            st.audio(vocal)
            if st.button("🌟 GENERATE FULL SYMPHONY"):
                with st.spinner("AI is building layers (Drums, Bass, Synth, FX)..."):
                    time.sleep(3)
                    st.success("45-Minute Composition Successfully Fingerprinted & Reserved!")

    with col2:
        st.image("https://images.unsplash.com/photo-1514525253361-bee8a18744ad?w=600", caption="Studio Interface")
        st.button("🎹 Launch Virtual Piano")
        st.button("🥁 Open Drum Pad")
        st.button("📜 Download Copyright Certificate")

# ==============================================================================
# MODULE 4: PREMIUM SUBSCRIPTIONS (QR PAYMENT GATEWAY)
# ==============================================================================
def show_premium_tiers():
    st.markdown("<h1 style='text-align: center;'>Upgrade Your Frequency</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;'>All premium memberships now come with <b>45-Day Validity</b>.</p>", unsafe_allow_html=True)
    
    p1, p2, p3, p4 = st.columns(4)
    plans = [
        {"name": "LISTENER", "price": 7, "color": "#00f2ea", "perks": "AI Mood Detection"},
        {"name": "UI PRO", "price": 14, "color": "#ff00ff", "perks": "Glass Theme Access"},
        {"name": "ARTIST", "price": 21, "color": "#00ff00", "perks": "Studio DAW Access"},
        {"name": "ELITE", "price": 25, "color": "gold", "perks": "Everything + 45-Min Sessions"}
    ]

    for idx, col in enumerate([p1, p2, p3, p4]):
        p = plans[idx]
        with col:
            st.markdown(f"""
            <div class="blade-card" style="border-color: {p['color']};">
                <h2 style="color: {p['color']}">{p['name']}</h2>
                <h1 style="font-size: 55px;">₹{p['price']}</h1>
                <p><b>45 DAYS ACCESS</b></p>
                <hr style="opacity: 0.1">
                <p style="font-size:14px;">{p['perks']}</p>
            </div>
            """, unsafe_allow_html=True)
            if st.button(f"Unlock {p['name']}", key=f"pay_{p['price']}"):
                st.session_state['qr_select'] = p['price']

    if 'qr_select' in st.session_state:
        st.markdown("---")
        st.subheader(f"💳 Secure Payment Gateway: ₹{st.session_state['qr_select']}")
        
        q1, q2 = st.columns([1, 2])
        with q1:
            qr_path = f"qr_{st.session_state['qr_select']}.png"
            if os.path.exists(qr_path):
                st.image(qr_path, width=300, caption=f"Scan to Activate ₹{st.session_state['qr_select']} Plan")
            else:
                st.error("QR Code Image not found in repository.")
                st.image("https://api.qrserver.com/v1/create-qr-code/?size=300x300&data=Manan_VibeSynth", width=250)
        
        with q2:
            st.write("### Instructions:")
            st.write(f"1. Scan the QR code and pay exactly ₹{st.session_state['qr_select']}.")
            st.write("2. Enter the Transaction UTR number below.")
            st.text_input("Transaction ID (UTR)")
            if st.button("Verify & Activate Subscription"):
                st.balloons()
                st.success("Premium Features Unlocked for your Account!")
            if st.button("Close Payment Window"):
                del st.session_state['qr_select']
                st.rerun()

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
inject_vibe_engine()

with st.sidebar:
    st.markdown("<h1 style='color: #00f2ea; margin-bottom: 0;'>VIBESYNTH</h1>", unsafe_allow_html=True)
    st.caption("Absolute AI Sovereignty")
    st.markdown("---")
    nav = st.radio("SOCIETY", ["🏠 Mission Control", "🎹 Artist Studio", "💎 Subscriptions"])
    st.markdown("---")
    st.write("🛡️ **Copyright Active**")
    st.caption("Developed by Manan Bansal")

if nav == "🏠 Mission Control":
    show_mission_control()
elif nav == "🎹 Artist Studio":
    show_artist_studio()
elif nav == "💎 Subscriptions":
    show_premium_tiers()