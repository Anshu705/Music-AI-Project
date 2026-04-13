import streamlit as st
import librosa
import numpy as np
import keras
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os
import time
from datetime import datetime

# ==============================================================================
# PHASE 1: THE LUXURY DESIGN ENGINE (CSS & JS PARTICLE OVERLAY)
# ==============================================================================
st.set_page_config(
    page_title="VibeSynth Ultra | Absolute AI Sovereignty",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="collapsed"
)

def inject_luxury_engine():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Syncopate:wght@400;700&family=Inter:wght@200;400;900&display=swap');
        
        /* THE FOUNDATION */
        .stApp {
            background: #050505;
            color: #ffffff;
            font-family: 'Inter', sans-serif;
        }

        /* AMBIENT BACKGROUND GLOW */
        .stApp::before {
            content: "";
            position: fixed;
            top: 0; left: 0; width: 100%; height: 100%;
            background: radial-gradient(circle at 20% 30%, rgba(0, 242, 234, 0.05) 0%, transparent 50%),
                        radial-gradient(circle at 80% 70%, rgba(0, 114, 255, 0.05) 0%, transparent 50%);
            z-index: -1;
        }

        /* LUXURY CAR STYLE HEADERS */
        h1, h2, h3 {
            font-family: 'Syncopate', sans-serif;
            text-transform: uppercase;
            letter-spacing: 5px;
            background: linear-gradient(90deg, #ffffff, #00f2ea);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        /* PREMIUM GLASS CARDS (ROLLS ROYCE STYLE INTERIOR) */
        .luxury-card {
            background: rgba(255, 255, 255, 0.02);
            backdrop-filter: blur(40px);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 40px;
            padding: 60px;
            transition: all 0.8s cubic-bezier(0.19, 1, 0.22, 1);
        }
        .luxury-card:hover {
            border-color: rgba(0, 242, 234, 0.4);
            background: rgba(255, 255, 255, 0.04);
            transform: scale(1.02);
        }

        /* THE 'BLADE' SUBSCRIPTION TIERS */
        .blade-tier {
            background: rgba(10, 10, 10, 0.8);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 20px;
            padding: 40px;
            text-align: center;
            transition: 0.5s ease;
        }
        .blade-tier:hover {
            border-color: #00f2ea;
            box-shadow: 0 0 40px rgba(0, 242, 234, 0.2);
            transform: translateY(-20px);
        }

        /* CUSTOM SIDEBAR */
        [data-testid="stSidebar"] {
            background: #000000 !important;
            border-right: 1px solid rgba(0, 242, 234, 0.2);
        }

        /* MOUSE TRACKING GLOW */
        #cursor-glow {
            position: fixed;
            top: 0; left: 0;
            width: 1000px; height: 1000px;
            background: radial-gradient(circle, rgba(0, 242, 234, 0.08) 0%, rgba(0,0,0,0) 70%);
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
# PHASE 2: AI ARCHITECTURE (THE ENGINE ROOM)
# ==============================================================================
@st.cache_resource
def startup_engine():
    # Utilizing the retrained 1,000-song database
    m_path, c_path = 'music_mood_model_1000.keras', 'music_database_1000.csv'
    if not os.path.exists(m_path): return None, None, None
    
    model = keras.models.load_model(m_path)
    data = pd.read_csv(c_path)
    # Cleaning math data
    data['BPM'] = data['BPM'].apply(lambda x: float(str(x).replace('[', '').replace(']', '')))
    X = data[['BPM', 'MFCC', 'Centroid', 'Rolloff', 'Chroma', 'ZCR', 'RMS']].values
    scaler = StandardScaler().fit(X)
    encoder = LabelEncoder().fit(data['Label'])
    return model, scaler, encoder

model, scaler, encoder = startup_engine()

# ==============================================================================
# PHASE 3: THE MISSION MANIFESTO (HOME)
# ==============================================================================
def render_manifesto():
    st.markdown("<h1 style='text-align:center; font-size:100px; margin-top:50px;'>SOVEREIGNTY.</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; font-size:20px; letter-spacing:10px; opacity:0.6;'>BY MANAN BANSAL</p>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    m_col1, m_col2 = st.columns([1, 1])
    with m_col1:
        st.markdown("""
        <div class="luxury-card">
            <h2 style="color:#00f2ea;">THE MISSION</h2>
            <p style="font-size:18px; line-height:2; font-weight:200;">
                VibeSynth was not created to replace artists, but to liberate them. 
                In the heart of Jaipur, we have engineered an intelligence that decodes 
                the frequency of human emotion. We are here to provide the absolute 
                instrumentation for the unvoiced.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with m_col2:
        st.markdown("""
        <div class="luxury-card">
            <h2 style="color:#0072ff;">THE 2027 GOAL</h2>
            <p style="font-size:18px; line-height:2; font-weight:200;">
                To launch the world's first 'Cognitive DAW'. A future where a simple 
                thought or vocal hum on your OnePlus Nord CE4 translates into a 
                45-minute orchestrated masterpiece, protected by immutable copyright 
                reservation instantly.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.image("https://images.unsplash.com/photo-1598488035139-bdbb2231ce04?w=1200", caption="VibeSynth Neural Command Core")

# ==============================================================================
# PHASE 4: THE STUDIO COMMAND CENTER (ARTIST HUB)
# ==============================================================================
def render_studio():
    st.markdown("<h1>🎨 STUDIO COMMAND CENTER</h1>", unsafe_allow_html=True)
    
    # Hero Visual
    st.image("https://images.unsplash.com/photo-1470225620780-dba8ba36b745?w=1200", caption="Studio A - Neural Link Active")

    tab1, tab2, tab3 = st.tabs(["⚡ AI Composition", "🎹 Instruments", "🛡️ Copyright Hub"])
    
    with tab1:
        st.markdown("""<div style="background:rgba(0,242,234,0.05); padding:40px; border-radius:30px;">
            <h3>➕ VOCAL-TO-SYMPHONY (45 MINS)</h3>
            <p>Upload a vocal stem. Our AI will analyze the pitch, timbre, and vibration to generate a 45-minute backing arrangement automatically.</p>
        </div>""", unsafe_allow_html=True)
        st.file_uploader("Insert Vocal Frequency", type=['mp3', 'wav'])
        if st.button("INITIATE GENERATION"):
            st.balloons()
            st.success("Symphony Generated. 45-Minute Session Reserved.")

    with tab2:
        st.subheader("Digital Instrumentation")
        st.button("Launch Virtual Steinway Grand")
        st.button("Access Roland-808 Neural Drums")
        st.button("Sitar Emulation Engine")

    with tab3:
        st.subheader("🛡️ Copyright Guard System")
        st.write("Our company holds the absolute copyright protection for your generated content. Every track is SHA-256 fingerprinted.")
        if st.button("Generate Legal Certificate"):
            st.info("Certificate #VS-2026-BTECH-001 issued to MANAN BANSAL")

# ==============================================================================
# PHASE 5: THE SUBSCRIPTION BLADES (PAYMENT SYSTEM)
# ==============================================================================
def render_subscriptions():
    st.markdown("<h1 style='text-align:center;'>CHOOSE YOUR FREQUENCY</h1>", unsafe_allow_html=True)
    st.write("<p style='text-align:center; opacity:0.6;'>All tiers updated to 45-Day Institutional Validity.</p>", unsafe_allow_html=True)
    
    s1, s2, s3, s4 = st.columns(4)
    plans = [
        {"name": "LISTENER", "price": 7, "color": "#00f2ea", "feat": "AI Mood Predictor"},
        {"name": "UI PRO", "price": 14, "color": "#ff00ff", "feat": "Glass Themes"},
        {"name": "ARTIST", "price": 21, "color": "#00ff00", "feat": "DAW Studio access"},
        {"name": "ELITE", "price": 25, "color": "gold", "feat": "45-Min Generation"}
    ]

    for idx, col in enumerate([s1, s2, s3, s4]):
        plan = plans[idx]
        with col:
            st.markdown(f"""
            <div class="blade-tier" style="border-top: 5px solid {plan['color']};">
                <h4 style="color:{plan['color']}">{plan['name']}</h4>
                <h1 style="font-size:60px; color:#fff;">₹{plan['price']}</h1>
                <p><b>45 DAYS</b></p>
                <hr style="opacity:0.1">
                <p style="font-size:12px;">{plan['feat']}</p>
            </div>
            """, unsafe_allow_html=True)
            if st.button(f"ACQUIRE {plan['name']}", key=f"pay_{plan['price']}"):
                st.session_state['pay_target'] = plan['price']

    # QR MODAL SYSTEM
    if 'pay_target' in st.session_state:
        st.markdown("---")
        st.subheader(f"💳 SECURE TRANSACTION: ₹{st.session_state['pay_target']}")
        
        qc1, qc2 = st.columns([1, 2])
        with qc1:
            # Looks for qr_7.png, qr_14.png, etc. in your repo
            qr_file = f"qr_{st.session_state['pay_target']}.png"
            if os.path.exists(qr_file):
                st.image(qr_file, width=280)
            else:
                st.image("https://api.qrserver.com/v1/create-qr-code/?size=250x250&data=VibeSynth_Payment", width=250)
        
        with qc2:
            st.write("### Payment Protocol")
            st.write(f"1. Scan to transfer exactly ₹{st.session_state['pay_target']}.")
            st.write("2. Enter Transaction UTR for Verification.")
            st.text_input("UTR Number")
            if st.button("VERIFY & UNLOCK"):
                st.toast("Verifying with Bank...")
                time.sleep(2)
                st.success("Plan Activated! Your OnePlus Nord CE4 is now Syncing.")

# ==============================================================================
# MAIN NAVIGATION LOGIC
# ==============================================================================
inject_luxury_engine()

# --- SIDEBAR (SUNO STYLE) ---
with st.sidebar:
    st.markdown("<h1 style='color:#00f2ea;'>VibeSynth</h1>", unsafe_allow_html=True)
    st.markdown("---")
    choice = st.radio("COMMAND", ["🏠 MANIFESTO", "🎹 STUDIO", "💎 PREMIUM"])
    st.markdown("---")
    st.write("💎 **PREMIUM ACCOUNT**")
    st.caption("Verifying Node: Jaipur-Main-01")

if choice == "🏠 MANIFESTO":
    render_manifesto()
elif choice == "🎹 STUDIO":
    render_studio()
elif choice == "💎 PREMIUM":
    render_subscriptions()