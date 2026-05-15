            import streamlit as st
import os
import time
import numpy as np
import pandas as pd
import tempfile
import random
from datetime import datetime

# ── Optional heavy deps ──────────────────────────────────────────────
try:
    import keras
    import librosa
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    BRAIN_AVAILABLE = True
except ImportError:
    BRAIN_AVAILABLE = False

try:
    from moviepy import AudioFileClip
    from pedalboard import Pedalboard, Reverb, LowShelfFilter, HighShelfFilter, Chorus, NoiseGate, Compressor
    from pedalboard.io import AudioFile
    PEDALBOARD_AVAILABLE = True
except ImportError:
    PEDALBOARD_AVAILABLE = False

# ════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="VibeSynth Ultra | Absolute AI Sovereignty",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ════════════════════════════════════════════════════════════════════
def init():
    d = {
        "logged_in": False, "user_name": "", "user_email": "",
        "user_plan": "FREE", "tracks": [], "cert_count": 0,
        "last_mood": None, "last_features": {},
        "drum_pattern": [[False]*16 for _ in range(6)],
        "drum_bpm": 120, "show_pay": False,
        "pay_amount": 0, "pay_plan": "",
    }
    for k, v in d.items():
        if k not in st.session_state:
            st.session_state[k] = v
init()

# ════════════════════════════════════════════════════════════════════
#  AI MODEL LOADER
# ════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_super_brain():
    if not BRAIN_AVAILABLE:
        return None, None, None
    model_path = 'music_mood_model_1000.keras'
    csv_path   = 'music_database_1000.csv'
    if not os.path.exists(model_path) or not os.path.exists(csv_path):
        return None, None, None
    model = keras.models.load_model(model_path)
    data  = pd.read_csv(csv_path)
    data['BPM'] = data['BPM'].apply(lambda x: float(str(x).replace('[','').replace(']','')))
    X = data[['BPM','MFCC','Centroid','Rolloff','Chroma','ZCR','RMS']].values
    scaler  = StandardScaler().fit(X)
    encoder = LabelEncoder().fit(data['Label'])
    return model, scaler, encoder

brain, scaler, encoder = load_super_brain()

def extract_and_predict(path, model, scaler, encoder):
    y, sr = librosa.load(path, duration=30)
    bpm, _ = librosa.beat.beat_track(y=y, sr=sr)
    bpm_val = float(bpm[0]) if isinstance(bpm, np.ndarray) else float(bpm)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20))
    cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    roll = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    chro = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))
    zcr  = np.mean(librosa.feature.zero_crossing_rate(y))
    rms  = np.mean(librosa.feature.rms(y=y))
    feats = np.array([[bpm_val, mfcc, cent, roll, chro, zcr, rms]])
    pred  = model.predict(scaler.transform(feats))
    mood  = encoder.inverse_transform([np.argmax(pred)])[0]
    return mood, bpm_val, zcr, rms, cent

def apply_live_effect(input_path, output_path):
    if not PEDALBOARD_AVAILABLE:
        return False
    tmp = input_path
    if input_path.endswith('.mp4'):
        clip = AudioFileClip(input_path)
        tmp  = input_path.replace('.mp4', '.wav')
        clip.write_audiofile(tmp, logger=None)
    board = Pedalboard([
        NoiseGate(threshold_db=-40.0, ratio=1.5, release_ms=250),
        Compressor(threshold_db=-15.0, ratio=2.5),
        Chorus(rate_hz=0.5, depth=0.1, centre_delay_ms=7.0, feedback=0.0, mix=0.3),
        LowShelfFilter(cutoff_frequency_hz=120, gain_db=4.0),
        HighShelfFilter(cutoff_frequency_hz=10000, gain_db=-2.0),
        Reverb(room_size=0.85, damping=0.4, wet_level=0.35, dry_level=0.8, width=1.0),
    ])
    with AudioFile(tmp) as f:
        with AudioFile(output_path, 'w', f.samplerate, f.num_channels) as o:
            while f.tell() < f.frames:
                chunk = f.read(f.samplerate)
                o.write(board(chunk, f.samplerate, reset=False))
    if input_path.endswith('.mp4'):
        os.remove(tmp)
    return True

# ════════════════════════════════════════════════════════════════════
#  FULL-SCREEN HTML (instruments, nav, home, pricing, auth)
# ════════════════════════════════════════════════════════════════════
FULL_HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<link href="https://fonts.googleapis.com/css2?family=Syncopate:wght@400;700&family=DM+Sans:wght@200;300;400;700&family=Space+Mono:wght@400;700&display=swap" rel="stylesheet">
<style>
/* ════ RESET ════ */
*{margin:0;padding:0;box-sizing:border-box}
:root{
  --c:#00f2ea;--c2:#0072ff;--gold:#f0c040;--rose:#ff3b6b;
  --bg:#040408;--bg2:#07070d;--surface:rgba(255,255,255,.03);
  --border:rgba(255,255,255,.07);--text:#e8e8f0;
  --muted:rgba(232,232,240,.42);
}
html{scroll-behavior:smooth}
body{background:var(--bg);color:var(--text);font-family:'DM Sans',sans-serif;overflow-x:hidden;cursor:none}

/* ── CURSOR ── */
#cur{position:fixed;width:10px;height:10px;background:var(--c);border-radius:50%;pointer-events:none;z-index:99999;mix-blend-mode:screen;transition:.1s}
#cur-r{position:fixed;width:32px;height:32px;border:1px solid rgba(0,242,234,.5);border-radius:50%;pointer-events:none;z-index:99998;transition:all .2s ease}

/* ── AMBIENT ── */
body::before{content:'';position:fixed;inset:0;
  background:radial-gradient(ellipse 70% 50% at 10% 15%,rgba(0,242,234,.07) 0,transparent 65%),
             radial-gradient(ellipse 60% 70% at 90% 85%,rgba(0,114,255,.07) 0,transparent 65%),
             radial-gradient(ellipse 40% 40% at 50% 50%,rgba(240,192,64,.03) 0,transparent 65%);
  z-index:-2;pointer-events:none}
body::after{content:'';position:fixed;inset:0;
  background-image:url("data:image/svg+xml,%3Csvg viewBox='0 0 300 300' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='.75' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='.04'/%3E%3C/svg%3E");
  opacity:.5;z-index:-1;pointer-events:none}

/* ── PAGES ── */
.page{display:none;animation:pgIn .4s ease}
.page.active{display:block}
@keyframes pgIn{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:none}}

/* ════ NAVBAR ════ */
nav{
  position:fixed;top:0;left:0;right:0;z-index:5000;
  display:flex;align-items:center;justify-content:space-between;
  padding:16px 52px;
  background:rgba(4,4,8,.82);backdrop-filter:blur(28px) saturate(1.4);
  border-bottom:1px solid var(--border);
}
.logo{font-family:'Syncopate',sans-serif;font-size:17px;letter-spacing:6px;text-transform:uppercase;
  background:linear-gradient(90deg,#fff,var(--c));-webkit-background-clip:text;-webkit-text-fill-color:transparent;cursor:pointer}
.nav-links{display:flex;gap:34px;align-items:center}
.nav-link{font-family:'Space Mono',monospace;font-size:9px;letter-spacing:3px;text-transform:uppercase;
  color:var(--muted);cursor:pointer;transition:color .3s;background:none;border:none;padding:4px}
.nav-link:hover,.nav-link.on{color:var(--c)!important}
.nav-cta{background:linear-gradient(135deg,var(--c),var(--c2));border:none;color:#000;
  font-family:'Syncopate',sans-serif;font-size:9px;letter-spacing:3px;text-transform:uppercase;
  padding:11px 24px;border-radius:40px;cursor:pointer;font-weight:700;transition:all .3s}
.nav-cta:hover{box-shadow:0 0 28px rgba(0,242,234,.5);transform:scale(1.04)}

/* ════ HERO ════ */
.hero{min-height:100vh;display:flex;flex-direction:column;align-items:center;justify-content:center;
  text-align:center;padding:120px 48px 80px;position:relative;overflow:hidden}
.hero-badge{display:inline-flex;align-items:center;gap:8px;border:1px solid rgba(0,242,234,.3);
  border-radius:40px;padding:7px 18px;font-family:'Space Mono',monospace;font-size:9px;letter-spacing:2px;
  color:var(--c);margin-bottom:48px;background:rgba(0,242,234,.05)}
.bdot{width:6px;height:6px;background:var(--c);border-radius:50%;animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.2}}
.hero-title{font-family:'Syncopate',sans-serif;font-size:clamp(72px,13vw,180px);line-height:.88;font-weight:700;
  background:linear-gradient(180deg,#fff 30%,rgba(255,255,255,.12));-webkit-background-clip:text;-webkit-text-fill-color:transparent;
  letter-spacing:-4px;margin-bottom:16px}
.hero-sub{font-family:'Syncopate',sans-serif;font-size:clamp(12px,1.8vw,17px);letter-spacing:14px;
  color:var(--c);text-transform:uppercase;margin-bottom:28px}
.hero-desc{max-width:540px;font-size:16px;line-height:1.9;color:var(--muted);margin-bottom:52px;font-weight:200}
.hero-cta{display:flex;gap:18px;justify-content:center;flex-wrap:wrap}
.btn-p{background:linear-gradient(135deg,var(--c),var(--c2));border:none;color:#000;
  padding:17px 44px;border-radius:60px;font-family:'Syncopate',sans-serif;font-size:10px;
  letter-spacing:4px;font-weight:700;cursor:pointer;transition:all .4s;text-transform:uppercase}
.btn-p:hover{box-shadow:0 0 50px rgba(0,242,234,.5);transform:translateY(-3px)}
.btn-g{background:transparent;border:1px solid rgba(255,255,255,.15);color:var(--text);
  padding:17px 44px;border-radius:60px;font-family:'Syncopate',sans-serif;font-size:10px;
  letter-spacing:4px;cursor:pointer;transition:all .4s;text-transform:uppercase}
.btn-g:hover{border-color:var(--c);color:var(--c);box-shadow:0 0 24px rgba(0,242,234,.12)}
.hero-stats{display:flex;gap:72px;margin-top:80px;padding-top:48px;border-top:1px solid var(--border);flex-wrap:wrap;justify-content:center}
.stt{text-align:center}
.stn{font-family:'Syncopate',sans-serif;font-size:44px;
  background:linear-gradient(90deg,var(--c),var(--c2));-webkit-background-clip:text;-webkit-text-fill-color:transparent;font-weight:700}
.stl{font-family:'Space Mono',monospace;font-size:9px;letter-spacing:3px;color:var(--muted);text-transform:uppercase;margin-top:5px}

/* ════ TICKER ════ */
.ticker{border-top:1px solid var(--border);border-bottom:1px solid var(--border);
  padding:15px 0;overflow:hidden;background:rgba(0,0,0,.45)}
.ticker-inner{display:flex;gap:72px;animation:tick 22s linear infinite;white-space:nowrap;width:max-content}
@keyframes tick{from{transform:translateX(0)}to{transform:translateX(-50%)}}
.ti{font-family:'Space Mono',monospace;font-size:10px;letter-spacing:3px;color:var(--muted);display:flex;align-items:center;gap:14px}
.ti span{color:var(--c)}

/* ════ MUSIC PLAYER BAR (Spotify-style) ════ */
.player-bar{
  position:fixed;bottom:0;left:0;right:0;z-index:4000;
  background:rgba(7,7,15,.96);backdrop-filter:blur(30px);
  border-top:1px solid var(--border);
  display:flex;align-items:center;justify-content:space-between;
  padding:12px 32px;gap:20px;
}
.pb-track{display:flex;align-items:center;gap:14px;min-width:200px}
.pb-art{width:48px;height:48px;border-radius:10px;
  background:linear-gradient(135deg,rgba(0,242,234,.25),rgba(0,114,255,.25));
  display:flex;align-items:center;justify-content:center;font-size:20px;flex-shrink:0}
.pb-name{font-size:13px;font-weight:400;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:160px}
.pb-artist{font-family:'Space Mono',monospace;font-size:9px;color:var(--muted);letter-spacing:1px;margin-top:2px}
.pb-heart{background:none;border:none;color:var(--muted);font-size:17px;cursor:pointer;transition:color .2s;margin-left:8px}
.pb-heart:hover,.pb-heart.on{color:var(--rose)}
.pb-center{flex:1;display:flex;flex-direction:column;align-items:center;gap:8px;max-width:580px}
.pb-controls{display:flex;align-items:center;gap:20px}
.pb-btn{background:none;border:none;color:var(--muted);font-size:20px;cursor:pointer;transition:color .2s}
.pb-btn:hover{color:var(--text)}
.pb-play{width:40px;height:40px;border-radius:50%;background:var(--c);border:none;color:#000;
  font-size:16px;cursor:pointer;display:flex;align-items:center;justify-content:center;
  transition:all .3s;flex-shrink:0}
.pb-play:hover{transform:scale(1.08);box-shadow:0 0 20px rgba(0,242,234,.5)}
.pb-progress{display:flex;align-items:center;gap:10px;width:100%}
.pb-time{font-family:'Space Mono',monospace;font-size:9px;color:var(--muted);letter-spacing:1px;flex-shrink:0}
.pb-bar{flex:1;height:3px;background:rgba(255,255,255,.12);border-radius:2px;cursor:pointer;position:relative}
.pb-fill{height:100%;border-radius:2px;background:linear-gradient(90deg,var(--c),var(--c2));transition:width .5s linear;position:relative}
.pb-fill::after{content:'';position:absolute;right:-5px;top:-4px;width:11px;height:11px;
  background:#fff;border-radius:50%;opacity:0;transition:opacity .2s;box-shadow:0 0 8px rgba(0,242,234,.6)}
.pb-bar:hover .pb-fill::after{opacity:1}
.pb-right{display:flex;align-items:center;gap:14px;min-width:180px;justify-content:flex-end}
.pb-vol-bar{width:80px;height:3px;background:rgba(255,255,255,.12);border-radius:2px;cursor:pointer}
.pb-vol-fill{height:100%;border-radius:2px;background:var(--c);width:70%}

/* ════ SECTIONS ════ */
.sec{padding:100px 60px}
.sec-label{font-family:'Space Mono',monospace;font-size:9px;letter-spacing:4px;color:var(--c);
  text-transform:uppercase;margin-bottom:14px;display:flex;align-items:center;gap:12px}
.sec-label::before{content:'';width:28px;height:1px;background:var(--c)}
.sec-title{font-family:'Syncopate',sans-serif;font-size:clamp(30px,5vw,58px);font-weight:700;
  line-height:1.05;background:linear-gradient(90deg,#fff,rgba(255,255,255,.5));
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:18px;letter-spacing:1px}
.sec-desc{font-size:16px;color:var(--muted);line-height:1.85;max-width:500px;font-weight:200}

/* ════ GLASS CARD ════ */
.glass{background:rgba(255,255,255,.025);border:1px solid var(--border);border-radius:24px;
  padding:40px;backdrop-filter:blur(20px);transition:all .5s cubic-bezier(.19,1,.22,1);position:relative;overflow:hidden}
.glass::before{content:'';position:absolute;top:0;left:15%;right:15%;height:1px;
  background:linear-gradient(90deg,transparent,rgba(255,255,255,.12),transparent)}
.glass:hover{border-color:rgba(0,242,234,.25);background:rgba(255,255,255,.04);transform:translateY(-7px);
  box-shadow:0 36px 70px rgba(0,0,0,.4),0 0 36px rgba(0,242,234,.04)}

/* ════ FEATURE GRID ════ */
.feat-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:18px;margin-top:56px}
.feat-icon{width:52px;height:52px;border-radius:14px;
  background:linear-gradient(135deg,rgba(0,242,234,.12),rgba(0,114,255,.12));
  border:1px solid rgba(0,242,234,.18);display:flex;align-items:center;justify-content:center;
  font-size:22px;margin-bottom:20px}
.feat-title{font-family:'Syncopate',sans-serif;font-size:12px;letter-spacing:2px;
  text-transform:uppercase;margin-bottom:10px;color:#fff}
.feat-desc{font-size:13px;color:var(--muted);line-height:1.8}

/* ════ BROWSE SHELF (Gaana / JioSaavn style) ════ */
.shelf{padding:0 60px 60px}
.shelf-head{display:flex;justify-content:space-between;align-items:center;margin-bottom:24px}
.shelf-title{font-family:'Syncopate',sans-serif;font-size:16px;letter-spacing:3px;color:#fff}
.shelf-more{font-family:'Space Mono',monospace;font-size:9px;letter-spacing:2px;color:var(--c);cursor:pointer;
  background:none;border:none}
.cards-row{display:flex;gap:16px;overflow-x:auto;padding-bottom:12px;scrollbar-width:none}
.cards-row::-webkit-scrollbar{display:none}
.mcard{flex-shrink:0;width:170px;cursor:pointer;transition:all .3s}
.mcard:hover{transform:translateY(-5px)}
.mcard-art{width:170px;height:170px;border-radius:16px;position:relative;overflow:hidden;
  background:linear-gradient(135deg,rgba(0,242,234,.15),rgba(0,114,255,.2));
  display:flex;align-items:center;justify-content:center;font-size:52px;margin-bottom:12px;
  border:1px solid var(--border)}
.mcard-art::after{content:'▶';position:absolute;bottom:10px;right:10px;
  width:36px;height:36px;border-radius:50%;background:var(--c);color:#000;
  display:flex;align-items:center;justify-content:center;font-size:12px;
  opacity:0;transition:opacity .3s;font-family:sans-serif}
.mcard:hover .mcard-art::after{opacity:1}
.mcard-name{font-size:13px;font-weight:400;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.mcard-sub{font-family:'Space Mono',monospace;font-size:9px;color:var(--muted);letter-spacing:1px;margin-top:3px}
.mcard-mood{display:inline-block;font-family:'Space Mono',monospace;font-size:7px;
  letter-spacing:2px;padding:2px 8px;border-radius:12px;border:1px solid rgba(0,242,234,.3);
  color:var(--c);margin-top:5px}

/* ════ GENRE PILLS (YouTube Music style) ════ */
.genre-pills{display:flex;gap:10px;flex-wrap:wrap;margin:0 60px 40px;padding-top:12px}
.pill{background:rgba(255,255,255,.06);border:1px solid var(--border);border-radius:40px;
  padding:9px 20px;font-family:'Space Mono',monospace;font-size:9px;letter-spacing:2px;
  color:var(--muted);cursor:pointer;transition:all .3s;text-transform:uppercase}
.pill.on,.pill:hover{background:rgba(0,242,234,.1);border-color:var(--c);color:var(--c)}

/* ════ INSTRUMENTS ════ */
.inst-sec{padding:80px 60px}
.inst-nav{display:flex;gap:10px;margin-bottom:40px;flex-wrap:wrap}
.ib{background:rgba(255,255,255,.04);border:1px solid var(--border);color:var(--muted);
  padding:11px 22px;border-radius:40px;cursor:pointer;font-family:'Space Mono',monospace;
  font-size:9px;letter-spacing:2px;text-transform:uppercase;transition:all .3s}
.ib.on,.ib:hover{background:rgba(0,242,234,.09);border-color:var(--c);color:var(--c)}
.isec{display:none}.isec.on{display:block}

/* DRUM */
.drum-wrap{background:rgba(0,0,0,.55);border:1px solid var(--border);border-radius:24px;padding:40px}
.dgrid{display:grid;grid-template-columns:90px repeat(16,1fr);gap:7px;align-items:center;margin-bottom:28px}
.dlabel{font-family:'Space Mono',monospace;font-size:9px;letter-spacing:2px;color:var(--muted);text-transform:uppercase}
.dpad{height:32px;border-radius:5px;background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.05);cursor:pointer;transition:all .1s}
.dpad:hover{background:rgba(0,242,234,.2);border-color:var(--c)}
.dpad.on{background:var(--c);border-color:var(--c);box-shadow:0 0 10px rgba(0,242,234,.6)}
.dpad.beat{background:var(--gold);box-shadow:0 0 14px rgba(240,192,64,.8)}
.dpad.sep{border-left:2px solid rgba(255,255,255,.15)}
.transport{display:flex;align-items:center;gap:18px;padding-top:24px;border-top:1px solid var(--border)}
.tbtn{width:44px;height:44px;border-radius:50%;border:1px solid var(--border);
  background:rgba(255,255,255,.04);color:var(--text);cursor:pointer;
  display:flex;align-items:center;justify-content:center;font-size:16px;transition:all .3s}
.tbtn:hover,.tbtn.pl{background:var(--c);border-color:var(--c);color:#000;box-shadow:0 0 18px rgba(0,242,234,.4)}
.bpm-ctrl{display:flex;align-items:center;gap:10px}
.bpm-lbl{font-family:'Space Mono',monospace;font-size:9px;letter-spacing:2px;color:var(--muted)}
.bpm-v{font-family:'Syncopate',sans-serif;font-size:26px;color:var(--c);min-width:56px;text-align:center}
.arr{width:26px;height:17px;background:rgba(255,255,255,.06);border:1px solid var(--border);
  border-radius:4px;cursor:pointer;display:flex;align-items:center;justify-content:center;
  font-size:9px;color:var(--muted);transition:all .2s}
.arr:hover{background:rgba(0,242,234,.2);color:var(--c)}

/* PIANO */
.piano-wrap{background:rgba(0,0,0,.65);border:1px solid var(--border);border-radius:24px;padding:40px;overflow-x:auto}
.pnd{font-family:'Space Mono',monospace;font-size:28px;color:var(--c);text-align:center;padding:14px;
  min-height:58px;letter-spacing:4px;text-shadow:0 0 18px rgba(0,242,234,.55)}
.piano{display:flex;position:relative;height:190px;width:max-content}
.wk{width:48px;height:190px;background:linear-gradient(180deg,#d4d8e0 0%,#fff 60%);
  border:1px solid #2a2a3e;border-radius:0 0 8px 8px;cursor:pointer;position:relative;
  flex-shrink:0;transition:background .08s;box-shadow:2px 3px 8px rgba(0,0,0,.35)}
.wk:hover{background:linear-gradient(180deg,#b0eee8,#d8fffc)}
.wk.dn{background:linear-gradient(180deg,var(--c),rgba(0,242,234,.5))!important}
.bk{width:28px;height:118px;background:linear-gradient(180deg,#0c0c18,#1a1a2a);
  position:absolute;z-index:2;border-radius:0 0 5px 5px;cursor:pointer;
  border:1px solid #000;transition:background .08s;box-shadow:3px 4px 10px rgba(0,0,0,.55)}
.bk:hover{background:linear-gradient(180deg,#003699,#0055cc)}
.bk.dn{background:linear-gradient(180deg,var(--c),var(--c2))!important}
.klbl{position:absolute;bottom:7px;left:50%;transform:translateX(-50%);
  font-family:'Space Mono',monospace;font-size:7px;color:rgba(0,0,0,.25);pointer-events:none}
.pcontrols{display:flex;gap:18px;margin-top:20px;align-items:center;flex-wrap:wrap}
.cg{display:flex;flex-direction:column;gap:7px}
.clbl{font-family:'Space Mono',monospace;font-size:8px;letter-spacing:2px;color:var(--muted);text-transform:uppercase}
.cslider{-webkit-appearance:none;width:130px;height:2px;background:rgba(255,255,255,.1);border-radius:2px;cursor:pointer}
.cslider::-webkit-slider-thumb{-webkit-appearance:none;width:13px;height:13px;background:var(--c);border-radius:50%;box-shadow:0 0 7px rgba(0,242,234,.5)}
.octd{font-family:'Syncopate',sans-serif;font-size:22px;color:var(--c)}
select.csel{background:rgba(255,255,255,.04);border:1px solid var(--border);color:var(--text);
  padding:7px 12px;border-radius:8px;font-family:'Space Mono',monospace;font-size:9px;letter-spacing:1px}

/* SITAR */
.sitar-wrap{background:rgba(0,0,0,.65);border:1px solid var(--border);border-radius:24px;padding:40px}
.sstr{display:flex;align-items:center;gap:18px;padding:16px 22px;border-radius:13px;
  background:rgba(255,255,255,.025);border:1px solid var(--border);cursor:pointer;
  margin-bottom:10px;transition:all .25s;user-select:none}
.sstr:hover{background:rgba(240,192,64,.06);border-color:rgba(240,192,64,.35);transform:translateX(4px)}
.sstr:active,.sstr.plk{background:rgba(240,192,64,.1)!important;border-color:rgba(240,192,64,.5)!important;box-shadow:0 0 18px rgba(240,192,64,.2)}
.snote{font-family:'Syncopate',sans-serif;font-size:13px;color:var(--gold);width:36px;flex-shrink:0}
.sline{flex:1;height:2px;background:linear-gradient(90deg,var(--gold),rgba(240,192,64,.15));border-radius:1px;transition:all .25s}
.sstr:hover .sline,.sstr.plk .sline{box-shadow:0 0 10px var(--gold);height:3px}
.shz{font-family:'Space Mono',monospace;font-size:9px;color:var(--muted)}
.sdesc{font-family:'Space Mono',monospace;font-size:9px;color:var(--muted);width:150px;flex-shrink:0}

/* SYNTH */
.synth-wrap{background:rgba(0,0,0,.65);border:1px solid var(--border);border-radius:24px;padding:40px}
.padgrid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:20px}
.spad{aspect-ratio:1;border-radius:14px;background:rgba(255,255,255,.04);border:1px solid var(--border);
  cursor:pointer;transition:all .15s;display:flex;flex-direction:column;
  align-items:center;justify-content:center;gap:6px;padding:10px;user-select:none}
.spad:hover{border-color:#0072ff;box-shadow:0 0 24px rgba(0,114,255,.3);background:rgba(0,114,255,.08)}
.spad:active,.spad.pl{background:rgba(0,242,234,.15)!important;border-color:var(--c)!important;box-shadow:0 0 30px rgba(0,242,234,.4)!important}
.spn{font-family:'Syncopate',sans-serif;font-size:16px;color:#fff}
.spc{font-family:'Space Mono',monospace;font-size:8px;color:var(--muted);letter-spacing:2px}

/* ════ PRICING ════ */
.pricing-sec{padding:100px 60px}
.pgrid{display:grid;grid-template-columns:repeat(4,1fr);gap:18px;margin-top:60px}
.pcard{background:rgba(7,7,15,.92);border:1px solid var(--border);border-radius:24px;
  padding:36px 28px;position:relative;overflow:hidden;transition:all .5s cubic-bezier(.19,1,.22,1)}
.pcard.featured{background:linear-gradient(180deg,rgba(0,242,234,.07),rgba(0,114,255,.07));border-color:rgba(0,242,234,.25)}
.pcard:hover{transform:translateY(-14px);box-shadow:0 36px 70px rgba(0,0,0,.5),0 0 36px rgba(0,242,234,.07);border-color:rgba(0,242,234,.35)}
.pbadge{font-family:'Space Mono',monospace;font-size:8px;letter-spacing:3px;text-transform:uppercase;
  display:inline-block;padding:3px 11px;border-radius:20px;margin-bottom:16px}
.pname{font-family:'Syncopate',sans-serif;font-size:19px;letter-spacing:3px;margin-bottom:20px}
.pprice{font-family:'Syncopate',sans-serif;font-size:50px;line-height:1;margin-bottom:3px}
.pcur{font-size:22px;vertical-align:top;margin-top:8px;display:inline-block}
.pper{font-family:'Space Mono',monospace;font-size:9px;letter-spacing:2px;color:var(--muted);margin-bottom:28px}
.pf{padding:8px 0;border-bottom:1px solid var(--border);font-size:12px;color:var(--muted);
  display:flex;align-items:center;gap:9px}
.pf::before{content:'→';color:var(--c);font-size:10px}
.pf:last-of-type{border:none}
.ptag{position:absolute;top:16px;right:16px;
  background:linear-gradient(90deg,var(--c),var(--c2));color:#000;
  font-family:'Syncopate',sans-serif;font-size:7px;letter-spacing:2px;
  padding:5px 12px;border-radius:20px;font-weight:700}
.pbtn{width:100%;padding:14px;border-radius:13px;font-family:'Syncopate',sans-serif;
  font-size:9px;letter-spacing:3px;font-weight:700;cursor:pointer;transition:all .3s;
  text-transform:uppercase;margin-top:24px}
.pbtn-o{background:transparent;border:1px solid rgba(255,255,255,.15);color:var(--text)}
.pbtn-o:hover{border-color:var(--c);color:var(--c)}
.pbtn-f{background:linear-gradient(135deg,var(--c),var(--c2));border:none;color:#000}
.pbtn-f:hover{box-shadow:0 0 26px rgba(0,242,234,.4)}

/* PAYMENT */
.pay-modal{display:none;margin-top:44px;max-width:580px;margin-left:auto;margin-right:auto}
.pay-modal.on{display:block}

/* ════ AUTH ════ */
.auth-page{min-height:100vh;display:flex;align-items:center;justify-content:center;padding:100px 20px 80px}
.auth-card{width:100%;max-width:420px;background:rgba(7,7,15,.96);border:1px solid var(--border);
  border-radius:28px;padding:52px 44px;backdrop-filter:blur(40px);position:relative;overflow:hidden}
.auth-card::before{content:'';position:absolute;top:0;left:20%;right:20%;height:1px;
  background:linear-gradient(90deg,transparent,var(--c),transparent)}
.atitle{font-family:'Syncopate',sans-serif;font-size:20px;letter-spacing:4px;text-align:center;margin-bottom:8px}
.asub{text-align:center;color:var(--muted);font-size:12px;margin-bottom:36px}
.aigrp{margin-bottom:18px}
.ailbl{font-family:'Space Mono',monospace;font-size:8px;letter-spacing:3px;color:var(--muted);
  text-transform:uppercase;display:block;margin-bottom:7px}
.aifield{width:100%;background:rgba(255,255,255,.04);border:1px solid var(--border);border-radius:11px;
  padding:14px 18px;color:var(--text);font-family:'DM Sans',sans-serif;font-size:13px;outline:none;transition:all .3s}
.aifield:focus{border-color:var(--c);box-shadow:0 0 18px rgba(0,242,234,.1);background:rgba(0,242,234,.03)}
.aifield::placeholder{color:rgba(255,255,255,.2)}
.abtn{width:100%;background:linear-gradient(135deg,var(--c),var(--c2));border:none;color:#000;
  padding:16px;border-radius:13px;font-family:'Syncopate',sans-serif;font-size:10px;
  letter-spacing:4px;font-weight:700;cursor:pointer;transition:all .4s;text-transform:uppercase;margin-top:6px}
.abtn:hover{box-shadow:0 0 36px rgba(0,242,234,.4);transform:translateY(-2px)}
.adiv{text-align:center;color:var(--muted);font-size:11px;margin:20px 0;display:flex;align-items:center;gap:14px}
.adiv::before,.adiv::after{content:'';flex:1;height:1px;background:var(--border)}
.alink{text-align:center;font-size:12px;color:var(--muted);margin-top:20px}
.alink a{color:var(--c);cursor:pointer;text-decoration:none}
.socbtn{width:100%;background:rgba(255,255,255,.04);border:1px solid var(--border);color:var(--text);
  padding:14px;border-radius:13px;display:flex;align-items:center;justify-content:center;gap:10px;
  cursor:pointer;transition:all .3s;font-size:12px;margin-bottom:10px}
.socbtn:hover{border-color:rgba(255,255,255,.2);background:rgba(255,255,255,.07)}

/* ════ DASHBOARD ════ */
.dash{padding:90px 0 80px;display:grid;grid-template-columns:260px 1fr;gap:0;min-height:100vh}
.dash-sb{background:rgba(0,0,0,.45);border-right:1px solid var(--border);padding:28px 20px;
  position:sticky;top:90px;height:fit-content}
.dav{width:70px;height:70px;border-radius:50%;
  background:linear-gradient(135deg,var(--c),var(--c2));
  display:flex;align-items:center;justify-content:center;
  font-family:'Syncopate',sans-serif;font-size:24px;color:#000;font-weight:700;
  margin:0 auto 14px;box-shadow:0 0 28px rgba(0,242,234,.3)}
.dname{font-family:'Syncopate',sans-serif;font-size:12px;letter-spacing:3px;text-align:center}
.demail{color:var(--muted);font-size:10px;margin-top:3px;text-align:center}
.dplan{display:inline-block;font-family:'Space Mono',monospace;font-size:7px;letter-spacing:2px;
  color:var(--gold);border:1px solid rgba(240,192,64,.3);padding:3px 10px;border-radius:20px;
  margin:8px auto 0;display:block;text-align:center;width:fit-content}
.dnav{margin-top:28px;display:flex;flex-direction:column;gap:6px}
.dni{display:flex;align-items:center;gap:10px;padding:12px 16px;border-radius:12px;
  cursor:pointer;transition:all .3s;font-size:9px;letter-spacing:2px;color:var(--muted);
  text-transform:uppercase;font-family:'Space Mono',monospace;border:1px solid transparent}
.dni:hover,.dni.on{background:rgba(0,242,234,.07);color:var(--c);border-color:rgba(0,242,234,.15)}
.dash-main{padding:36px 48px;display:flex;flex-direction:column;gap:22px}
.dhead{background:linear-gradient(135deg,rgba(0,242,234,.07),rgba(0,114,255,.07));
  border:1px solid rgba(0,242,234,.14);border-radius:20px;padding:36px;
  display:flex;justify-content:space-between;align-items:center}
.dgreet{font-family:'Syncopate',sans-serif;font-size:24px;letter-spacing:3px}
.ddate{font-family:'Space Mono',monospace;font-size:9px;color:var(--muted);letter-spacing:2px;margin-top:6px}
.sgrid{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}
.sc{background:rgba(255,255,255,.025);border:1px solid var(--border);border-radius:18px;padding:24px;transition:all .4s}
.sc:hover{border-color:rgba(0,242,234,.3);transform:translateY(-3px)}
.scn{font-family:'Syncopate',sans-serif;font-size:34px;
  background:linear-gradient(90deg,var(--c),var(--c2));-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.scl{font-size:9px;color:var(--muted);letter-spacing:2px;text-transform:uppercase;margin-top:6px}
.rtracks{background:rgba(255,255,255,.025);border:1px solid var(--border);border-radius:20px;padding:28px}
.rt-title{font-family:'Syncopate',sans-serif;font-size:12px;letter-spacing:3px;text-transform:uppercase;
  color:rgba(255,255,255,.6);margin-bottom:20px}
.titem{display:flex;align-items:center;gap:16px;padding:13px 0;border-bottom:1px solid rgba(255,255,255,.05);cursor:pointer;transition:padding .25s}
.titem:hover{padding-left:8px}
.titem:last-child{border:none}
.tthumb{width:44px;height:44px;border-radius:11px;background:linear-gradient(135deg,rgba(0,242,234,.18),rgba(0,114,255,.18));
  display:flex;align-items:center;justify-content:center;font-size:18px;flex-shrink:0}
.tname{font-size:13px}
.tmeta{font-family:'Space Mono',monospace;font-size:8px;color:var(--muted);letter-spacing:1px;margin-top:2px}
.tmood{font-family:'Space Mono',monospace;font-size:8px;letter-spacing:2px;padding:3px 9px;
  border-radius:20px;border:1px solid rgba(0,242,234,.3);color:var(--c);flex-shrink:0}

/* ════ MOOD BARS ════ */
.mbars{display:flex;align-items:flex-end;gap:5px;height:48px;margin-top:12px}
.mbar{width:7px;border-radius:3px;background:linear-gradient(180deg,var(--c),var(--c2));
  animation:ba 1.2s ease infinite;transform-origin:bottom}
@keyframes ba{0%,100%{transform:scaleY(.2)}50%{transform:scaleY(1)}}

/* ════ TOAST ════ */
.toast{position:fixed;bottom:90px;right:36px;z-index:9999;
  background:rgba(7,7,15,.96);border:1px solid rgba(0,242,234,.4);border-radius:14px;
  padding:18px 26px;font-family:'Space Mono',monospace;font-size:11px;letter-spacing:2px;color:var(--c);
  transform:translateY(80px);opacity:0;transition:all .35s cubic-bezier(.19,1,.22,1);
  backdrop-filter:blur(20px);box-shadow:0 18px 50px rgba(0,0,0,.5)}
.toast.show{transform:translateY(0);opacity:1}

/* ════ SCROLLBAR ════ */
::-webkit-scrollbar{width:5px}
::-webkit-scrollbar-track{background:var(--bg)}
::-webkit-scrollbar-thumb{background:rgba(0,242,234,.28);border-radius:3px}

/* ════ FOOTER ════ */
.footer{border-top:1px solid var(--border);padding:56px 60px 40px;
  display:grid;grid-template-columns:1.4fr 1fr 1fr 1fr;gap:48px}
.fl-logo{font-family:'Syncopate',sans-serif;font-size:15px;letter-spacing:6px;text-transform:uppercase;
  background:linear-gradient(90deg,#fff,var(--c));-webkit-background-clip:text;-webkit-text-fill-color:transparent;display:block;margin-bottom:14px}
.fl-desc{font-size:12px;color:var(--muted);line-height:1.8;max-width:220px}
.fc h4{font-family:'Syncopate',sans-serif;font-size:10px;letter-spacing:3px;text-transform:uppercase;
  color:rgba(255,255,255,.35);margin-bottom:16px}
.fc ul{list-style:none}
.fc ul li{margin-bottom:9px}
.fc ul li a{font-size:12px;color:var(--muted);cursor:pointer;transition:color .3s;text-decoration:none}
.fc ul li a:hover{color:var(--c)}
.fbot{border-top:1px solid var(--border);padding:22px 60px;
  display:flex;justify-content:space-between;align-items:center}
.fcp{font-family:'Space Mono',monospace;font-size:8px;color:rgba(232,232,240,.22);letter-spacing:2px}

/* ════ MISSION ════ */
.mgrid{display:grid;grid-template-columns:1fr 1fr;gap:22px;margin-top:56px}

@media(max-width:960px){
  .feat-grid{grid-template-columns:1fr 1fr}
  .pgrid{grid-template-columns:1fr 1fr}
  .sgrid{grid-template-columns:1fr 1fr}
  .dash{grid-template-columns:1fr}
  .footer{grid-template-columns:1fr 1fr;gap:28px}
  nav{padding:14px 22px}
  .sec{padding:72px 24px}
}
</style>
</head>
<body>

<div id="cur"></div><div id="cur-r"></div>
<div class="toast" id="toast"></div>

<!-- ── NAV ── -->
<nav>
  <div class="logo" onclick="go('home')">VibeSynth</div>
  <div class="nav-links">
    <button class="nav-link on" onclick="go('home')">Studio</button>
    <button class="nav-link" onclick="go('browse')">Browse</button>
    <button class="nav-link" onclick="go('instruments')">Instruments</button>
    <button class="nav-link" onclick="go('pricing')">Premium</button>
    <button class="nav-link" id="dnl" style="display:none" onclick="go('dashboard')">Dashboard</button>
    <button class="nav-link" id="lol" style="display:none" onclick="logout()">Logout</button>
  </div>
  <button class="nav-cta" id="nacta" onclick="go('login')">Enter Studio</button>
</nav>

<!-- ════════ PAGE: HOME ════════ -->
<div class="page active" id="pg-home">
<section class="hero">
  <div class="hero-badge"><div class="bdot"></div>AI Music Intelligence · Jaipur, India</div>
  <h1 class="hero-title">VIBE<br>SYNTH</h1>
  <p class="hero-sub">Absolute AI Sovereignty</p>
  <p class="hero-desc">The world's first cognitive music intelligence. Decode emotion. Compose reality. Claim your frequency — engineered from the heart of Jaipur.</p>
  <div class="hero-cta">
    <button class="btn-p" onclick="go('signup')">Begin Creation</button>
    <button class="btn-g" onclick="go('instruments')">Explore Studio</button>
  </div>
  <div class="hero-stats">
    <div class="stt"><div class="stn">1K+</div><div class="stl">Songs Analyzed</div></div>
    <div class="stt"><div class="stn">7</div><div class="stl">Mood Classes</div></div>
    <div class="stt"><div class="stn">45</div><div class="stl">Min Generation</div></div>
    <div class="stt"><div class="stn">∞</div><div class="stl">Possibilities</div></div>
  </div>
</section>

<div class="ticker"><div class="ticker-inner" id="ticker"></div></div>

<!-- Genre Pills -->
<div class="genre-pills">
  <div class="pill on" onclick="setPill(this,'All')">All</div>
  <div class="pill" onclick="setPill(this,'Happy')">😊 Happy</div>
  <div class="pill" onclick="setPill(this,'Sad')">😢 Sad</div>
  <div class="pill" onclick="setPill(this,'Energetic')">⚡ Energetic</div>
  <div class="pill" onclick="setPill(this,'Calm')">🌊 Calm</div>
  <div class="pill" onclick="setPill(this,'Aggressive')">🔥 Aggressive</div>
  <div class="pill" onclick="setPill(this,'Relaxed')">🌙 Relaxed</div>
  <div class="pill" onclick="setPill(this,'Classical')">🎻 Classical</div>
  <div class="pill" onclick="setPill(this,'Bollywood')">🎬 Bollywood</div>
  <div class="pill" onclick="setPill(this,'Electronic')">🎛️ Electronic</div>
</div>

<!-- Trending shelf -->
<div class="shelf">
  <div class="shelf-head">
    <div class="shelf-title">🔥 Trending Now</div>
    <button class="shelf-more">See all →</button>
  </div>
  <div class="cards-row" id="trending-row"></div>
</div>

<!-- Recently Played -->
<div class="shelf">
  <div class="shelf-head">
    <div class="shelf-title">⏱ Recently Played</div>
    <button class="shelf-more">See all →</button>
  </div>
  <div class="cards-row" id="recent-row"></div>
</div>

<!-- Feature grid -->
<div class="sec" style="padding-top:20px">
  <div class="sec-label">What We Offer</div>
  <div class="sec-title">The Full Arsenal</div>
  <div class="feat-grid">
    <div class="glass" style="padding:36px">
      <div class="feat-icon">🧠</div>
      <div class="feat-title">Mood Classification</div>
      <div class="feat-desc">7-class neural emotion detection. Keras model trained on 1,000 songs with production-grade accuracy.</div>
    </div>
    <div class="glass" style="padding:36px">
      <div class="feat-icon">🎹</div>
      <div class="feat-title">Virtual Instruments</div>
      <div class="feat-desc">Studio-grade piano, drum machine, sitar emulation and synth pads — all in-browser, zero latency.</div>
    </div>
    <div class="glass" style="padding:36px">
      <div class="feat-icon">⚡</div>
      <div class="feat-title">Vocal to Symphony</div>
      <div class="feat-desc">Hum a melody, receive a 45-minute orchestrated masterpiece. The 2027 Cognitive DAW is coming.</div>
    </div>
    <div class="glass" style="padding:36px">
      <div class="feat-icon">🛡️</div>
      <div class="feat-title">Copyright Guard</div>
      <div class="feat-desc">SHA-256 fingerprinting on every track. Immutable certificates issued to your account instantly.</div>
    </div>
    <div class="glass" style="padding:36px">
      <div class="feat-icon">💎</div>
      <div class="feat-title">Elite Subscription</div>
      <div class="feat-desc">Four tiers of creative power. 45-day institutional validity. Upgrade anytime via UPI.</div>
    </div>
    <div class="glass" style="padding:36px">
      <div class="feat-icon">🌐</div>
      <div class="feat-title">Jaipur-Main Node</div>
      <div class="feat-desc">Engineered in the Pink City. Ultra-low latency inference. Built for global creators.</div>
    </div>
  </div>
</div>

<!-- Mission -->
<div class="sec" style="padding-top:0">
  <div class="sec-label">Our Purpose</div>
  <div class="sec-title">Mission &amp; Goal</div>
  <div class="mgrid">
    <div class="glass">
      <div style="font-size:28px;margin-bottom:16px">🎯</div>
      <div class="feat-title" style="color:var(--c);margin-bottom:12px">The Mission</div>
      <p style="font-size:14px;line-height:1.9;color:var(--muted);font-weight:200">VibeSynth was not created to replace artists, but to liberate them. In the heart of Jaipur, we engineered an intelligence that decodes the frequency of human emotion. We exist to provide absolute instrumentation for the unvoiced.</p>
    </div>
    <div class="glass">
      <div style="font-size:28px;margin-bottom:16px">🚀</div>
      <div class="feat-title" style="color:var(--c2);margin-bottom:12px">The 2027 Goal</div>
      <p style="font-size:14px;line-height:1.9;color:var(--muted);font-weight:200">To launch the world's first Cognitive DAW. A future where a simple thought or vocal hum translates into a 45-minute orchestrated masterpiece, protected by immutable copyright reservation instantly — on any device.</p>
    </div>
  </div>
  <div style="text-align:center;margin-top:60px">
    <div class="sec-label" style="justify-content:center;margin-bottom:12px">Created By</div>
    <div style="font-family:'Syncopate',sans-serif;font-size:clamp(26px,5vw,56px);letter-spacing:10px;background:linear-gradient(90deg,#f0c040,#00f2ea);-webkit-background-clip:text;-webkit-text-fill-color:transparent">MANAN BANSAL</div>
  </div>
</div>

<!-- Footer -->
<footer class="footer">
  <div><span class="fl-logo">VibeSynth</span><p class="fl-desc">AI-powered music intelligence from Jaipur. Decoding the frequency of human emotion since 2024.</p></div>
  <div class="fc"><h4>Studio</h4><ul><li><a onclick="go('instruments')">Instruments</a></li><li><a>AI Composition</a></li><li><a>Copyright Hub</a></li><li><a>Mood Engine</a></li></ul></div>
  <div class="fc"><h4>Plans</h4><ul><li><a onclick="go('pricing')">Listener ₹7</a></li><li><a onclick="go('pricing')">UI Pro ₹14</a></li><li><a onclick="go('pricing')">Artist ₹21</a></li><li><a onclick="go('pricing')">Elite ₹25</a></li></ul></div>
  <div class="fc"><h4>Account</h4><ul><li><a onclick="go('login')">Sign In</a></li><li><a onclick="go('signup')">Create Account</a></li><li><a onclick="go('dashboard')">Dashboard</a></li></ul></div>
</footer>
<div class="fbot">
  <span class="fcp">© 2026 VibeSynth Ultra · Jaipur-Main-01 · By Manan Bansal</span>
  <span class="fcp">All frequencies reserved.</span>
</div>
</div>

<!-- ════════ PAGE: BROWSE ════════ -->
<div class="page" id="pg-browse">
<div class="sec">
  <div class="sec-label">Discover</div>
  <div class="sec-title">Browse All</div>
  <div class="genre-pills" style="margin:28px 0 40px">
    <div class="pill on">All Moods</div>
    <div class="pill">😊 Happy</div><div class="pill">😢 Sad</div>
    <div class="pill">⚡ Energetic</div><div class="pill">🌊 Calm</div>
    <div class="pill">🔥 Aggressive</div><div class="pill">🌙 Relaxed</div>
    <div class="pill">😰 Fearful</div>
  </div>
  <div class="shelf" style="padding:0"><div class="shelf-head"><div class="shelf-title">🎵 For You</div><button class="shelf-more">Shuffle ↺</button></div><div class="cards-row" id="browse-row1"></div></div>
  <br>
  <div class="shelf" style="padding:0"><div class="shelf-head"><div class="shelf-title">🌐 Global Charts</div><button class="shelf-more">See all →</button></div><div class="cards-row" id="browse-row2"></div></div>
  <br>
  <div class="shelf" style="padding:0"><div class="shelf-head"><div class="shelf-title">🪕 Indian Ragas</div><button class="shelf-more">Explore →</button></div><div class="cards-row" id="browse-row3"></div></div>
</div>
</div>

<!-- ════════ PAGE: INSTRUMENTS ════════ -->
<div class="page" id="pg-instruments">
<div class="inst-sec">
  <div class="sec-label">Studio Command Center</div>
  <div class="sec-title">Virtual Instruments</div>
  <p style="color:var(--muted);font-size:14px;margin-bottom:40px">Click an instrument. Play with mouse or keyboard.</p>
  <div class="inst-nav">
    <button class="ib on" onclick="openInst('drum',this)">🥁 Drum Machine</button>
    <button class="ib" onclick="openInst('piano',this)">🎹 Steinway Grand</button>
    <button class="ib" onclick="openInst('sitar',this)">🪕 Sitar</button>
    <button class="ib" onclick="openInst('synth',this)">🎛️ Synth Pads</button>
  </div>

  <!-- DRUM -->
  <div class="isec on" id="is-drum">
  <div class="drum-wrap">
    <div class="sec-label" style="margin-bottom:24px">Roland-808 Neural Drums</div>
    <div id="dgc"></div>
    <div class="transport">
      <button class="tbtn" id="pbtn" onclick="togglePlay()">▶</button>
      <button class="tbtn" onclick="clearDrum()">⬛</button>
      <button class="tbtn" onclick="randomPattern()" title="Random">🎲</button>
      <div class="bpm-ctrl">
        <span class="bpm-lbl">BPM</span>
        <div style="display:flex;flex-direction:column;gap:3px">
          <div class="arr" onclick="bpmChange(1)">▲</div>
          <div class="arr" onclick="bpmChange(-1)">▼</div>
        </div>
        <span class="bpm-v" id="bpmv">120</span>
      </div>
      <div style="margin-left:auto;font-family:'Space Mono',monospace;font-size:9px;color:var(--muted);letter-spacing:2px" id="beatc">STEP 1/16</div>
    </div>
  </div>
  </div>

  <!-- PIANO -->
  <div class="isec" id="is-piano">
  <div class="piano-wrap">
    <div class="sec-label" style="margin-bottom:8px">Steinway Grand Neural Piano</div>
    <div class="pnd" id="pnd">— PLAY A KEY —</div>
    <div class="piano" id="pkeys"></div>
    <div class="pcontrols">
      <div class="cg"><span class="clbl">Volume</span><input type="range" class="cslider" min="0" max="100" value="80" oninput="pVol=this.value/100"></div>
      <div class="cg"><span class="clbl">Reverb</span><input type="range" class="cslider" min="0" max="100" value="30" oninput="pRev=this.value/100"></div>
      <div class="cg"><span class="clbl">Octave</span>
        <div style="display:flex;align-items:center;gap:10px">
          <div class="arr" style="width:32px;height:26px" onclick="changeOct(-1)">◀</div>
          <span class="octd" id="octd">4</span>
          <div class="arr" style="width:32px;height:26px" onclick="changeOct(1)">▶</div>
        </div>
      </div>
      <div class="cg"><span class="clbl">Waveform</span>
        <select class="csel" onchange="pWave=this.value">
          <option value="sine">Sine</option><option value="triangle">Triangle</option>
          <option value="sawtooth">Sawtooth</option><option value="square">Square</option>
        </select>
      </div>
    </div>
    <div style="margin-top:14px;font-family:'Space Mono',monospace;font-size:8px;color:var(--muted);letter-spacing:2px">
      KEYS: A S D F G H J · W E T Y U (sharps) · OCTAVE <span id="oct-hint">4</span>
    </div>
  </div>
  </div>

  <!-- SITAR -->
  <div class="isec" id="is-sitar">
  <div class="sitar-wrap">
    <div class="sec-label" style="margin-bottom:8px">Sitar Emulation Engine</div>
    <p style="color:var(--muted);font-size:12px;margin-bottom:24px">Click string to pluck · Keys 1–7</p>
    <div id="sstrings"></div>
    <div style="margin-top:28px">
      <div class="sec-label" style="margin-bottom:14px">Quick Raga</div>
      <div style="display:flex;gap:10px;flex-wrap:wrap">
        <button class="ib" onclick="playRaga([261.63,277.18,329.63,349.23,392,415.30,493.88])">Bhairav (Dawn)</button>
        <button class="ib" onclick="playRaga([261.63,293.66,329.63,369.99,392,440,493.88])">Yaman (Evening)</button>
        <button class="ib" onclick="playRaga([261.63,277.18,311.13,349.23,392,415.30,466.16])">Bhairavi (Dusk)</button>
        <button class="ib" onclick="playRaga([261.63,293.66,329.63,349.23,392,440,493.88])">Kafi (Midnight)</button>
      </div>
    </div>
  </div>
  </div>

  <!-- SYNTH PADS -->
  <div class="isec" id="is-synth">
  <div class="synth-wrap">
    <div class="sec-label" style="margin-bottom:8px">Neural Synth Pads</div>
    <p style="color:var(--muted);font-size:12px;margin-bottom:18px">Hold pad for sustained tone · Keys 1–9</p>
    <div style="display:flex;gap:18px;align-items:center;flex-wrap:wrap;margin-bottom:20px">
      <div class="cg"><span class="clbl">Waveform</span>
        <select class="csel" id="stype">
          <option value="sine">Sine</option><option value="triangle">Triangle</option>
          <option value="sawtooth">Sawtooth</option><option value="square">Square</option>
        </select>
      </div>
      <div class="cg"><span class="clbl">Attack ms</span><input type="range" class="cslider" min="10" max="500" value="50" id="satk"></div>
      <div class="cg"><span class="clbl">Release ms</span><input type="range" class="cslider" min="100" max="3000" value="900" id="srel"></div>
    </div>
    <div class="padgrid" id="padgrid"></div>
  </div>
  </div>

  <!-- Copyright banner -->
  <div style="margin-top:40px">
  <div class="glass" style="display:flex;align-items:center;gap:28px;flex-wrap:wrap;padding:30px 36px">
    <div style="font-size:32px">🛡️</div>
    <div style="flex:1"><div class="feat-title" style="margin-bottom:6px">Copyright Guard System</div>
      <p style="color:var(--muted);font-size:13px;line-height:1.8">Every track SHA-256 fingerprinted. Immutable certificates issued instantly.</p></div>
    <button class="btn-p" onclick="genCert()">Generate Certificate</button>
  </div>
  </div>
</div>
</div>

<!-- ════════ PAGE: PRICING ════════ -->
<div class="page" id="pg-pricing">
<div class="pricing-sec">
  <div style="text-align:center;margin-bottom:14px">
    <div class="sec-label" style="justify-content:center">Choose Your Frequency</div>
    <div class="sec-title" style="text-align:center">Premium Access</div>
    <p style="color:var(--muted);font-size:14px;max-width:460px;margin:0 auto;line-height:1.8">All tiers include 45-day institutional validity. One-time UPI activation.</p>
  </div>
  <div class="pgrid">
    <div class="pcard">
      <div class="pbadge" style="color:#00f2ea;border:1px solid rgba(0,242,234,.3);background:rgba(0,242,234,.05)">Listener</div>
      <div class="pname">Listener</div>
      <div class="pprice"><span class="pcur">₹</span>7</div>
      <div class="pper">per 45 days</div>
      <div class="pf">AI Mood Predictor</div><div class="pf">7 Mood Classes</div>
      <div class="pf">1,000 Song Database</div><div class="pf">Audio Analysis</div><div class="pf">Basic Dashboard</div>
      <button class="pbtn pbtn-o" onclick="initPay(7,'LISTENER')">Acquire Access</button>
    </div>
    <div class="pcard">
      <div class="pbadge" style="color:#ff00ff;border:1px solid rgba(255,0,255,.3);background:rgba(255,0,255,.05)">UI Pro</div>
      <div class="pname">UI Pro</div>
      <div class="pprice"><span class="pcur">₹</span>14</div>
      <div class="pper">per 45 days</div>
      <div class="pf">Everything in Listener</div><div class="pf">Glass Themes</div>
      <div class="pf">Custom Cursor Effects</div><div class="pf">Virtual Instruments</div><div class="pf">Export Sessions</div>
      <button class="pbtn pbtn-o" onclick="initPay(14,'UI PRO')">Acquire Access</button>
    </div>
    <div class="pcard featured">
      <div class="ptag">Most Popular</div>
      <div class="pbadge" style="color:#00ff88;border:1px solid rgba(0,255,136,.3);background:rgba(0,255,136,.05)">Artist</div>
      <div class="pname">Artist</div>
      <div class="pprice"><span class="pcur">₹</span>21</div>
      <div class="pper">per 45 days</div>
      <div class="pf">Everything in UI Pro</div><div class="pf">Full DAW Studio</div>
      <div class="pf">Vocal-to-Symphony</div><div class="pf">SHA-256 Copyright</div><div class="pf">Legal Certificates</div>
      <button class="pbtn pbtn-f" onclick="initPay(21,'ARTIST')">Acquire Access</button>
    </div>
    <div class="pcard">
      <div class="pbadge" style="color:#f0c040;border:1px solid rgba(240,192,64,.3);background:rgba(240,192,64,.05)">Elite</div>
      <div class="pname" style="color:#f0c040">Elite</div>
      <div class="pprice" style="color:#f0c040"><span class="pcur" style="color:#f0c040">₹</span>25</div>
      <div class="pper">per 45 days</div>
      <div class="pf">Everything in Artist</div><div class="pf">45-Min Generation</div>
      <div class="pf">Priority Node Access</div><div class="pf">Dedicated Support</div><div class="pf">Early DAW Access</div>
      <button class="pbtn pbtn-o" style="border-color:#f0c040;color:#f0c040" onclick="initPay(25,'ELITE')">Acquire Elite</button>
    </div>
  </div>

  <!-- Payment Modal -->
  <div class="glass pay-modal" id="paymod">
    <div class="sec-label" style="margin-bottom:20px">💳 Secure Transaction</div>
    <div style="display:grid;grid-template-columns:1fr 1.4fr;gap:28px;align-items:start">
      <div>
        <img id="payqr" src="" style="border-radius:14px;width:100%;border:1px solid var(--border)">
        <p style="text-align:center;margin-top:10px;font-family:'Space Mono',monospace;font-size:8px;letter-spacing:2px;color:var(--muted)">SCAN WITH ANY UPI APP</p>
      </div>
      <div>
        <div style="font-family:'Syncopate',sans-serif;font-size:38px;color:var(--c);margin-bottom:6px" id="payamt">₹0</div>
        <div style="font-family:'Space Mono',monospace;font-size:9px;color:var(--muted);letter-spacing:2px;margin-bottom:20px" id="payplan">PLAN</div>
        <p style="color:var(--muted);font-size:12px;margin-bottom:18px;line-height:1.8">1. Scan QR or transfer to UPI ID<br>2. Enter UTR number below<br>3. Click Verify to activate</p>
        <div class="aigrp"><label class="ailbl">UTR / Transaction ID</label>
          <input class="aifield" type="text" id="utrf" placeholder="12-digit UTR number"></div>
        <button class="btn-p" style="width:100%;padding:14px" onclick="verifyPay()">Verify &amp; Unlock</button>
        <button onclick="document.getElementById('paymod').classList.remove('on')" style="margin-top:12px;width:100%;background:none;border:1px solid var(--border);color:var(--muted);padding:12px;border-radius:12px;cursor:pointer;font-family:'Space Mono',monospace;font-size:9px;letter-spacing:2px">CANCEL</button>
      </div>
    </div>
  </div>
</div>
</div>

<!-- ════════ PAGE: DASHBOARD ════════ -->
<div class="page" id="pg-dashboard">
<div class="dash">
  <div class="dash-sb">
    <div class="dav" id="dav">MB</div>
    <div class="dname" id="dname">MANAN BANSAL</div>
    <div class="demail" id="demail">manan@vibesynth.io</div>
    <div class="dplan" id="dplan">ELITE MEMBER</div>
    <div class="dnav">
      <div class="dni on"><span>📊</span>Overview</div>
      <div class="dni" onclick="go('browse')"><span>🎵</span>Browse Music</div>
      <div class="dni" onclick="go('instruments')"><span>🎹</span>Studio</div>
      <div class="dni" onclick="go('pricing')"><span>💎</span>Upgrade</div>
      <div class="dni" onclick="genCert()"><span>🛡️</span>Copyright</div>
      <div class="dni" onclick="logout()"><span>🚪</span>Logout</div>
    </div>
  </div>
  <div class="dash-main">
    <div class="dhead">
      <div>
        <div class="dgreet" id="dgreet">Good Evening, Manan</div>
        <div class="ddate" id="ddate"></div>
      </div>
      <button class="btn-p" onclick="go('instruments')">Open Studio ↗</button>
    </div>
    <div class="sgrid">
      <div class="sc"><div class="scn">47</div><div class="scl">Tracks Analyzed</div></div>
      <div class="sc"><div class="scn">12</div><div class="scl">Sessions</div></div>
      <div class="sc"><div class="scn" id="dcc">3</div><div class="scl">Certificates</div></div>
      <div class="sc"><div class="scn">31</div><div class="scl">Days Left</div></div>
    </div>
    <div class="rtracks">
      <div class="rt-title">Recent Tracks</div>
      <div class="titem"><div class="tthumb">🎵</div><div style="flex:1"><div class="tname">Neural Sunrise · Session 1</div><div class="tmeta">Piano · 3:42 · Apr 12 2026</div></div><div class="tmood">Happy</div></div>
      <div class="titem"><div class="tthumb">🥁</div><div style="flex:1"><div class="tname">808 Pressure · Trap Loop</div><div class="tmeta">Drums · 1:28 · Apr 11 2026</div></div><div class="tmood">Aggressive</div></div>
      <div class="titem"><div class="tthumb">🪕</div><div style="flex:1"><div class="tname">Raga Bhairav · Evening</div><div class="tmeta">Sitar · 6:15 · Apr 10 2026</div></div><div class="tmood">Calm</div></div>
      <div class="titem"><div class="tthumb">🎛️</div><div style="flex:1"><div class="tname">Synth Pad Experiment v3</div><div class="tmeta">Synth · 2:00 · Apr 9 2026</div></div><div class="tmood">Energetic</div></div>
    </div>
    <div class="glass">
      <div class="rt-title">Mood Analytics · Last 7 Days</div>
      <div style="display:flex;gap:10px;align-items:flex-end;height:110px;margin-top:8px" id="mchart"></div>
      <div style="display:flex;gap:14px;flex-wrap:wrap;margin-top:16px" id="mleg"></div>
    </div>
  </div>
</div>
</div>

<!-- ════════ PAGE: LOGIN ════════ -->
<div class="page" id="pg-login">
<div class="auth-page">
<div class="auth-card">
  <div class="logo" style="text-align:center;display:block;margin-bottom:22px;font-size:14px" onclick="go('home')">VibeSynth</div>
  <div class="atitle">Welcome Back</div>
  <div class="asub">Sign in to your creative sovereignty</div>
  <button class="socbtn"><svg width="17" height="17" viewBox="0 0 18 18"><path fill="#4285F4" d="M16.51 8H8.98v3h4.3c-.18 1-.74 1.48-1.6 2.04v2.01h2.6a7.8 7.8 0 0 0 2.38-5.88c0-.57-.05-.66-.15-1.18z"/><path fill="#34A853" d="M8.98 17c2.16 0 3.97-.72 5.3-1.94l-2.6-2a4.8 4.8 0 0 1-7.18-2.54H1.83v2.07A8 8 0 0 0 8.98 17z"/><path fill="#FBBC05" d="M4.5 10.52a4.8 4.8 0 0 1 0-3.04V5.41H1.83a8 8 0 0 0 0 7.18l2.67-2.07z"/><path fill="#EA4335" d="M8.98 4.18c1.17 0 2.23.4 3.06 1.2l2.3-2.3A8 8 0 0 0 1.83 5.4L4.5 7.49a4.77 4.77 0 0 1 4.48-3.3z"/></svg>Continue with Google</button>
  <div class="adiv">or sign in with email</div>
  <div class="aigrp"><label class="ailbl">Email Address</label><input class="aifield" type="email" id="lei" placeholder="you@vibesynth.io"></div>
  <div class="aigrp"><label class="ailbl">Password</label><input class="aifield" type="password" id="lpi" placeholder="••••••••"></div>
  <button class="abtn" onclick="doLogin()">Enter Studio</button>
  <div class="alink">No account? <a onclick="go('signup')">Create one</a></div>
</div>
</div>
</div>

<!-- ════════ PAGE: SIGNUP ════════ -->
<div class="page" id="pg-signup">
<div class="auth-page">
<div class="auth-card">
  <div class="logo" style="text-align:center;display:block;margin-bottom:22px;font-size:14px" onclick="go('home')">VibeSynth</div>
  <div class="atitle">Create Account</div>
  <div class="asub">Join the movement. Claim your frequency.</div>
  <button class="socbtn"><svg width="17" height="17" viewBox="0 0 18 18"><path fill="#4285F4" d="M16.51 8H8.98v3h4.3c-.18 1-.74 1.48-1.6 2.04v2.01h2.6a7.8 7.8 0 0 0 2.38-5.88c0-.57-.05-.66-.15-1.18z"/><path fill="#34A853" d="M8.98 17c2.16 0 3.97-.72 5.3-1.94l-2.6-2a4.8 4.8 0 0 1-7.18-2.54H1.83v2.07A8 8 0 0 0 8.98 17z"/><path fill="#FBBC05" d="M4.5 10.52a4.8 4.8 0 0 1 0-3.04V5.41H1.83a8 8 0 0 0 0 7.18l2.67-2.07z"/><path fill="#EA4335" d="M8.98 4.18c1.17 0 2.23.4 3.06 1.2l2.3-2.3A8 8 0 0 0 1.83 5.4L4.5 7.49a4.77 4.77 0 0 1 4.48-3.3z"/></svg>Sign up with Google</button>
  <div class="adiv">or create with email</div>
  <div class="aigrp"><label class="ailbl">Full Name</label><input class="aifield" type="text" id="sni" placeholder="Manan Bansal"></div>
  <div class="aigrp"><label class="ailbl">Email Address</label><input class="aifield" type="email" id="sei" placeholder="you@vibesynth.io"></div>
  <div class="aigrp"><label class="ailbl">Password</label><input class="aifield" type="password" id="spi" placeholder="Min. 8 characters"></div>
  <button class="abtn" onclick="doSignup()">Create Account</button>
  <div class="alink">Already have one? <a onclick="go('login')">Sign in</a></div>
</div>
</div>
</div>

<!-- ════════ PLAYER BAR ════════ -->
<div class="player-bar" id="pbar">
  <div class="pb-track">
    <div class="pb-art" id="pb-art">🎵</div>
    <div>
      <div class="pb-name" id="pb-name">Neural Sunrise</div>
      <div class="pb-artist" id="pb-artist">VibeSynth Studio</div>
    </div>
    <button class="pb-heart" id="pb-heart" onclick="toggleHeart()">♡</button>
  </div>
  <div class="pb-center">
    <div class="pb-controls">
      <button class="pb-btn" title="Shuffle">⇄</button>
      <button class="pb-btn" onclick="prevTrack()">⏮</button>
      <button class="pb-play" id="pb-play" onclick="togglePBPlay()">▶</button>
      <button class="pb-btn" onclick="nextTrack()">⏭</button>
      <button class="pb-btn" title="Repeat">↻</button>
    </div>
    <div class="pb-progress">
      <span class="pb-time" id="pb-cur">0:00</span>
      <div class="pb-bar" onclick="seekBar(event)">
        <div class="pb-fill" id="pb-fill" style="width:0%"></div>
      </div>
      <span class="pb-time" id="pb-dur">3:42</span>
    </div>
  </div>
  <div class="pb-right">
    <button class="pb-btn" title="Queue">☰</button>
    <button class="pb-btn" title="Volume">🔊</button>
    <div class="pb-vol-bar"><div class="pb-vol-fill" id="pb-vf"></div></div>
  </div>
</div>

<script>
// ════ CURSOR ════
const cur=document.getElementById('cur'),curR=document.getElementById('cur-r');
let mx=0,my=0,rx=0,ry=0;
document.addEventListener('mousemove',e=>{
  mx=e.clientX;my=e.clientY;
  cur.style.left=mx+'px';cur.style.top=my+'px';
});
(function anim(){rx+=(mx-rx)*.1;ry+=(my-ry)*.1;curR.style.left=rx+'px';curR.style.top=ry+'px';requestAnimationFrame(anim)})();

// ════ TOAST ════
function toast(m){const t=document.getElementById('toast');t.textContent=m;t.classList.add('show');setTimeout(()=>t.classList.remove('show'),3200)}

// ════ TICKER ════
const tItems=['🎵 AI MOOD ENGINE','⚡ 1000+ SONGS','🎹 VIRTUAL STUDIO','🛡️ SHA-256 COPYRIGHT','🪕 SITAR ENGINE','🥁 808 NEURAL DRUMS','💎 ELITE ACCESS','🚀 2027 COGNITIVE DAW','🌐 JAIPUR-MAIN-01','🎬 BOLLYWOOD AI','🌊 RAGA INTELLIGENCE'];
const tic=document.getElementById('ticker');
if(tic){const r=[...tItems,...tItems].map(t=>`<span class="ti"><span>◆</span>${t}</span>`).join('');tic.innerHTML=r+r}

// ════ PAGE NAV ════
let cu=null;
function go(p){
  document.querySelectorAll('.page').forEach(x=>x.classList.remove('active'));
  document.getElementById('pg-'+p).classList.add('active');
  document.querySelectorAll('.nav-link').forEach(x=>x.classList.remove('on'));
  window.scrollTo(0,0);
  if(p==='dashboard')updateDash();
}

// ════ GENRE PILLS ════
function setPill(el,g){document.querySelectorAll('.pill').forEach(p=>p.classList.remove('on'));el.classList.add('on');toast('🎵 Showing: '+g)}

// ════ MUSIC CARDS ════
const TRACKS=[
  {e:'🎵',n:'Neural Sunrise',a:'VibeSynth Studio',m:'Happy',c:'#00f2ea'},
  {e:'🥁',n:'808 Pressure',a:'Manan Bansal',m:'Energetic',c:'#f0c040'},
  {e:'🪕',n:'Raga Bhairav',a:'Classical AI',m:'Calm',c:'#0072ff'},
  {e:'🎛️',n:'Synth Experiment',a:'VibeSynth',m:'Aggressive',c:'#ff3b6b'},
  {e:'🎹',n:'Moonlight Session',a:'Neural Grand',m:'Sad',c:'#8844ff'},
  {e:'🎸',n:'Jaipur Jam',a:'VibeSynth Collective',m:'Happy',c:'#44ff88'},
  {e:'🎺',n:'Midnight Frequency',a:'Manan Bansal',m:'Relaxed',c:'#ff8844'},
  {e:'🎻',n:'String Theory',a:'Classical AI',m:'Calm',c:'#00f2ea'},
];
function makeCard(t,idx){
  return`<div class="mcard" onclick="playTrack(${idx})">
    <div class="mcard-art" style="background:linear-gradient(135deg,${t.c}22,${t.c}44)">${t.e}</div>
    <div class="mcard-name">${t.n}</div>
    <div class="mcard-sub">${t.a}</div>
    <span class="mcard-mood">${t.m}</span>
  </div>`;
}
function buildShelves(){
  const ids=['trending-row','recent-row','browse-row1','browse-row2','browse-row3'];
  ids.forEach(id=>{
    const el=document.getElementById(id);
    if(el)el.innerHTML=[...TRACKS].sort(()=>Math.random()-.5).map((t,i)=>makeCard(t,i)).join('');
  });
}
buildShelves();

// ════ PLAYER BAR ════
let pbPlaying=false,pbProgress=0,pbInterval=null,pbHeartOn=false,curTrackIdx=0;
function playTrack(i){
  curTrackIdx=i;const t=TRACKS[i];
  document.getElementById('pb-name').textContent=t.n;
  document.getElementById('pb-artist').textContent=t.a;
  document.getElementById('pb-art').textContent=t.e;
  document.getElementById('pb-art').style.background=`linear-gradient(135deg,${t.c}33,${t.c}55)`;
  pbProgress=0;
  if(!pbPlaying){togglePBPlay();}
  else{startProgress();}
  toast('▶ Playing: '+t.n);
}
function togglePBPlay(){
  pbPlaying=!pbPlaying;
  document.getElementById('pb-play').textContent=pbPlaying?'⏸':'▶';
  pbPlaying?startProgress():clearInterval(pbInterval);
}
function startProgress(){
  clearInterval(pbInterval);
  const dur=222;// 3:42 in seconds
  pbInterval=setInterval(()=>{
    pbProgress=Math.min(100,pbProgress+(100/dur/10));
    document.getElementById('pb-fill').style.width=pbProgress+'%';
    const s=Math.floor(pbProgress/100*dur);
    document.getElementById('pb-cur').textContent=`${Math.floor(s/60)}:${String(s%60).padStart(2,'0')}`;
    if(pbProgress>=100){clearInterval(pbInterval);pbProgress=0;nextTrack();}
  },100);
}
function seekBar(e){const r=e.currentTarget.getBoundingClientRect();pbProgress=(e.clientX-r.left)/r.width*100;document.getElementById('pb-fill').style.width=pbProgress+'%';}
function nextTrack(){playTrack((curTrackIdx+1)%TRACKS.length)}
function prevTrack(){playTrack((curTrackIdx+TRACKS.length-1)%TRACKS.length)}
function toggleHeart(){pbHeartOn=!pbHeartOn;const h=document.getElementById('pb-heart');h.textContent=pbHeartOn?'♥':'♡';h.classList.toggle('on',pbHeartOn);toast(pbHeartOn?'❤️ Added to Liked':'Removed from Liked')}

// ════ AUTH ════
function doLogin(){
  const e=document.getElementById('lei').value,p=document.getElementById('lpi').value;
  if(!e||!p){toast('⚠️ Please fill all fields');return}
  cu={name:e.split('@')[0].replace(/\./g,' ').toUpperCase(),email:e,plan:'ARTIST'};
  afterLogin();
}
function doSignup(){
  const n=document.getElementById('sni').value,e=document.getElementById('sei').value,p=document.getElementById('spi').value;
  if(!n||!e||!p){toast('⚠️ Please fill all fields');return}
  if(p.length<8){toast('⚠️ Password must be 8+ chars');return}
  cu={name:n.toUpperCase(),email:e,plan:'LISTENER'};
  afterLogin();
}
function afterLogin(){
  document.getElementById('nacta').style.display='none';
  document.getElementById('dnl').style.display='inline';
  document.getElementById('lol').style.display='inline';
  toast('✅ Welcome to VibeSynth · '+cu.name);
  go('dashboard');
}
function logout(){
  cu=null;
  document.getElementById('nacta').style.display='inline-block';
  document.getElementById('dnl').style.display='none';
  document.getElementById('lol').style.display='none';
  toast('👋 Logged out');go('home');
}
function updateDash(){
  if(!cu)return;
  const ini=cu.name.split(' ').map(w=>w[0]).join('').slice(0,2);
  document.getElementById('dav').textContent=ini;
  document.getElementById('dname').textContent=cu.name;
  document.getElementById('demail').textContent=cu.email;
  document.getElementById('dplan').textContent=cu.plan+' MEMBER';
  const h=new Date().getHours();
  const g=h<12?'Good Morning':h<18?'Good Afternoon':'Good Evening';
  document.getElementById('dgreet').textContent=g+', '+cu.name.split(' ')[0];
  document.getElementById('ddate').textContent=new Date().toLocaleDateString('en-IN',{weekday:'long',year:'numeric',month:'long',day:'numeric'}).toUpperCase();
  buildMoodChart();
}
function buildMoodChart(){
  const days=['Mon','Tue','Wed','Thu','Fri','Sat','Sun'];
  const clrs=['#00f2ea','#0072ff','#f0c040','#8844ff','#ff4444','#44ff88','#ff8844'];
  const moods=['Happy','Calm','Energy','Sad','Aggr','Relax','Fear'];
  const vals=moods.map(()=>Math.floor(Math.random()*14+2));
  const mx=Math.max(...vals);
  const mc=document.getElementById('mchart');
  const ml=document.getElementById('mleg');
  if(!mc)return;
  mc.innerHTML=vals.map((v,i)=>`<div style="flex:1;display:flex;flex-direction:column;align-items:center;gap:6px">
    <div style="width:100%;border-radius:4px 4px 0 0;background:linear-gradient(180deg,${clrs[i]},${clrs[i]}44);height:${(v/mx*100)}px"></div>
    <span style="font-family:'Space Mono',monospace;font-size:8px;color:var(--muted);letter-spacing:1px">${days[i]}</span>
  </div>`).join('');
  ml.innerHTML=moods.map((m,i)=>`<div style="display:flex;align-items:center;gap:5px"><div style="width:7px;height:7px;border-radius:2px;background:${clrs[i]}"></div><span style="font-family:'Space Mono',monospace;font-size:8px;color:var(--muted);letter-spacing:1px">${m}</span></div>`).join('');
}

// ════ CERT ════
let certN=3;
function genCert(){
  const id='VS-'+Date.now().toString(36).toUpperCase();
  const nm=cu?cu.name:'MANAN BANSAL';
  certN++;
  const dc=document.getElementById('dcc');if(dc)dc.textContent=certN;
  toast('🛡️ Certificate '+id+' issued to '+nm);
}

// ════ PAYMENT ════
function initPay(a,p){
  document.getElementById('payamt').textContent='₹'+a;
  document.getElementById('payplan').textContent=p+' · 45 DAYS';
  document.getElementById('payqr').src=`https://api.qrserver.com/v1/create-qr-code/?size=220x220&data=VibeSynth_UPI_${a}&bgcolor=040408&color=00f2ea&margin=10`;
  document.getElementById('paymod').classList.add('on');
  document.getElementById('paymod').scrollIntoView({behavior:'smooth'});
}
function verifyPay(){
  const u=document.getElementById('utrf').value;
  if(!u||u.length<8){toast('⚠️ Enter valid UTR');return}
  toast('⏳ Verifying...');
  setTimeout(()=>{toast('✅ Plan Activated!');document.getElementById('paymod').classList.remove('on');if(cu){cu.plan='ELITE';go('dashboard');}else go('signup')},2500);
}

// ════ AUDIO CTX ════
let AC=null;
function getAC(){if(!AC)AC=new(window.AudioContext||window.webkitAudioContext)();if(AC.state==='suspended')AC.resume();return AC}

// ════ DRUM MACHINE ════
const DT=[
  {n:'KICK',f:55,t:'sine',c:'#00f2ea'},
  {n:'SNARE',f:200,t:'triangle',c:'#0072ff'},
  {n:'HI-HAT',f:8000,t:'square',c:'#f0c040'},
  {n:'CLAP',f:600,t:'square',c:'#ff3b6b'},
  {n:'TOM HI',f:120,t:'sine',c:'#44ff88'},
  {n:'TOM LO',f:80,t:'sine',c:'#ff8844'},
];
let dp=DT.map(()=>Array(16).fill(false)),bpm=120,playing=false,step=0,dint=null;
function buildDrum(){
  const c=document.getElementById('dgc');if(!c)return;
  let h='<div class="dgrid">';
  DT.forEach((t,ti)=>{
    h+=`<div class="dlabel" style="color:${t.c}">${t.n}</div>`;
    for(let s=0;s<16;s++){
      const sep=s===8?' sep':'';
      h+=`<div class="dpad${sep}" id="d_${ti}_${s}" onclick="togPad(${ti},${s})"></div>`;
    }
  });
  h+='</div>';c.innerHTML=h;
}
function togPad(ti,s){dp[ti][s]=!dp[ti][s];document.getElementById(`d_${ti}_${s}`).classList.toggle('on',dp[ti][s])}
function clearDrum(){dp=DT.map(()=>Array(16).fill(false));document.querySelectorAll('.dpad').forEach(p=>{p.classList.remove('on','beat')})}
function randomPattern(){
  clearDrum();
  DT.forEach((_,ti)=>{for(let s=0;s<16;s++)if(Math.random()>.65){dp[ti][s]=true;document.getElementById(`d_${ti}_${s}`).classList.add('on')}});
}
function togglePlay(){playing=!playing;const b=document.getElementById('pbtn');b.classList.toggle('pl',playing);b.textContent=playing?'⏹':'▶';playing?startDrum():stopDrum()}
function startDrum(){
  const st=60/bpm/4*1000;
  dint=setInterval(()=>{
    document.querySelectorAll('.dpad.beat').forEach(p=>p.classList.remove('beat'));
    DT.forEach((t,ti)=>{
      const el=document.getElementById(`d_${ti}_${step}`);
      if(el)el.classList.add('beat');
      if(dp[ti][step])hitDrum(t);
    });
    document.getElementById('beatc').textContent=`STEP ${step+1}/16`;
    step=(step+1)%16;
  },st);
}
function stopDrum(){clearInterval(dint);document.querySelectorAll('.dpad.beat').forEach(p=>p.classList.remove('beat'));step=0;document.getElementById('beatc').textContent='STEP 1/16'}
function hitDrum(t){
  const ac=getAC(),o=ac.createOscillator(),g=ac.createGain();
  o.connect(g);g.connect(ac.destination);
  o.type=t.t;o.frequency.setValueAtTime(t.f,ac.currentTime);
  o.frequency.exponentialRampToValueAtTime(.01,ac.currentTime+.1);
  g.gain.setValueAtTime(.8,ac.currentTime);
  g.gain.exponentialRampToValueAtTime(.001,ac.currentTime+.18);
  o.start();o.stop(ac.currentTime+.22);
}
function bpmChange(d){bpm=Math.min(200,Math.max(60,bpm+d));document.getElementById('bpmv').textContent=bpm;if(playing){stopDrum();startDrum()}}
buildDrum();

// ════ PIANO ════
const NOTES=['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'];
const WN=['C','D','E','F','G','A','B'];
const KMAP={'a':'C','w':'C#','s':'D','e':'D#','d':'E','f':'F','t':'F#','g':'G','y':'G#','h':'A','u':'A#','j':'B','k':'C+'};
let pOct=4,pVol=.8,pRev=.3,pWave='sine',pOscs={};
function noteFreq(n,o){return 261.63*Math.pow(2,(NOTES.indexOf(n)+(o-4)*12)/12)}
function buildPiano(){
  const c=document.getElementById('pkeys');if(!c)return;c.innerHTML='';
  let lo=0;
  for(let o=pOct;o<=pOct+1;o++){
    WN.forEach((n,wi)=>{
      const k=document.createElement('div');k.className='wk';
      k.dataset.note=n;k.dataset.oct=o;
      const lb=document.createElement('div');lb.className='klbl';lb.textContent=n+o;k.appendChild(lb);
      k.addEventListener('mousedown',()=>startPNote(n+o,noteFreq(n,o),k));
      k.addEventListener('mouseup',()=>stopPNote(n+o,k));
      k.addEventListener('mouseleave',()=>stopPNote(n+o,k));
      c.appendChild(k);
      const bp={'C':0,'D':1,'F':3,'G':4,'A':5};
      if(n in bp){
        const bk=document.createElement('div');bk.className='bk';
        const bn=n+'#';bk.dataset.note=bn;bk.dataset.oct=o;
        bk.style.left=(lo+(wi+.65)*48-14)+'px';
        bk.addEventListener('mousedown',e=>{e.stopPropagation();startPNote(bn+o,noteFreq(bn,o),bk)});
        bk.addEventListener('mouseup',()=>stopPNote(bn+o,bk));
        bk.addEventListener('mouseleave',()=>stopPNote(bn+o,bk));
        c.appendChild(bk);
      }
    });
    lo+=WN.length*48;
  }
}
function startPNote(id,freq,el){
  const ac=getAC();if(pOscs[id])return;
  const o=ac.createOscillator(),g=ac.createGain();
  o.type=pWave;o.frequency.value=freq;
  g.gain.setValueAtTime(0,ac.currentTime);
  g.gain.linearRampToValueAtTime(pVol*.4,ac.currentTime+.018);
  o.connect(g);g.connect(ac.destination);o.start();
  pOscs[id]={o,g};el.classList.add('dn');
  document.getElementById('pnd').textContent=id;
}
function stopPNote(id,el){
  if(!pOscs[id])return;
  const{o,g}=pOscs[id];const ac=getAC();
  g.gain.setTargetAtTime(0,ac.currentTime,.09);
  setTimeout(()=>{try{o.stop()}catch(e){}},300);
  delete pOscs[id];el.classList.remove('dn');
}
function changeOct(d){pOct=Math.min(6,Math.max(2,pOct+d));document.getElementById('octd').textContent=pOct;document.getElementById('oct-hint').textContent=pOct;buildPiano()}
document.addEventListener('keydown',e=>{
  if(document.getElementById('pg-instruments').classList.contains('active')&&document.getElementById('is-piano').classList.contains('on')){
    const nm=KMAP[e.key.toLowerCase()];
    if(nm&&!e.repeat){
      const isPlus=nm.endsWith('+');const oct=isPlus?pOct+1:pOct;const n=nm.replace('+','');
      const freq=noteFreq(n,oct);
      document.querySelectorAll(`[data-note="${n}"][data-oct="${oct}"]`).forEach(k=>startPNote(n+oct,freq,k));
    }
  }
  if(document.getElementById('is-sitar').classList.contains('on')){const si=parseInt(e.key)-1;if(si>=0&&si<SITAR.length)pluckS(si)}
  if(document.getElementById('is-synth').classList.contains('on')){const pi=parseInt(e.key)-1;if(pi>=0&&pi<PADS.length)trigPad(pi)}
});
document.addEventListener('keyup',e=>{
  const nm=KMAP[e.key.toLowerCase()];
  if(nm){const oct=nm.endsWith('+')?pOct+1:pOct;const n=nm.replace('+','');
    document.querySelectorAll(`[data-note="${n}"][data-oct="${oct}"]`).forEach(k=>stopPNote(n+oct,k));}
});

// ════ SITAR ════
const SITAR=[
  {n:'Sa',hz:261.63,d:'C · Root / Shadja'},
  {n:'Re',hz:293.66,d:'D · Rishabha'},
  {n:'Ga',hz:329.63,d:'E · Gandhara'},
  {n:'Ma',hz:349.23,d:'F · Madhyama'},
  {n:'Pa',hz:392.00,d:'G · Panchama'},
  {n:'Dha',hz:440.00,d:'A · Dhaivata'},
  {n:'Ni',hz:493.88,d:'B · Nishada'},
];
function buildSitar(){
  const c=document.getElementById('sstrings');if(!c)return;
  c.innerHTML=SITAR.map((s,i)=>`
  <div class="sstr" id="ss_${i}" onclick="pluckS(${i})">
    <div class="snote">${s.n}</div>
    <div class="sline"></div>
    <div class="shz">${s.hz.toFixed(2)} Hz</div>
    <div class="sdesc">${s.d}</div>
  </div>`).join('');
}
function pluckS(i){
  const s=SITAR[i];const ac=getAC();
  const o1=ac.createOscillator(),o2=ac.createOscillator(),g=ac.createGain();
  o1.type='sawtooth';o1.frequency.value=s.hz;
  o2.type='triangle';o2.frequency.value=s.hz*1.008;
  g.gain.setValueAtTime(.55,ac.currentTime);
  g.gain.exponentialRampToValueAtTime(.001,ac.currentTime+2.6);
  o1.connect(g);o2.connect(g);g.connect(ac.destination);
  o1.start();o2.start();o1.stop(ac.currentTime+3);o2.stop(ac.currentTime+3);
  const el=document.getElementById('ss_'+i);el.classList.add('plk');
  setTimeout(()=>el.classList.remove('plk'),2600);
}
function playRaga(freqs){freqs.forEach((f,i)=>setTimeout(()=>{const ac=getAC();const o=ac.createOscillator(),g=ac.createGain();o.type='sawtooth';o.frequency.value=f;g.gain.setValueAtTime(.4,ac.currentTime);g.gain.exponentialRampToValueAtTime(.001,ac.currentTime+1.5);o.connect(g);g.connect(ac.destination);o.start();o.stop(ac.currentTime+1.8)},i*300));toast('🪕 Playing Raga')}
buildSitar();

// ════ SYNTH PADS ════
const PADS=[
  {n:'Am',f:220,c:'Am7'},{n:'C',f:261.63,c:'Cmaj7'},{n:'Em',f:164.81,c:'Em7'},{n:'G',f:196,c:'Gmaj'},
  {n:'Dm',f:146.83,c:'Dm7'},{n:'F',f:174.61,c:'Fadd9'},{n:'Bm',f:246.94,c:'Bm7'},{n:'E',f:164.81,c:'Emaj'},
  {n:'A',f:220,c:'Asus2'},{n:'Fm',f:174.61,c:'Fm7'},{n:'Bb',f:233.08,c:'Bbmaj7'},{n:'D',f:293.66,c:'Dadd9'},
];
let padOscs={};
function buildPads(){
  const c=document.getElementById('padgrid');if(!c)return;
  c.innerHTML=PADS.map((p,i)=>`
  <div class="spad" id="sp_${i}" onmousedown="startPad(${i})" onmouseup="stopPad(${i})" onmouseleave="stopPad(${i})">
    <div class="spn">${p.n}</div>
    <div class="spc">${p.c}</div>
  </div>`).join('');
}
function startPad(i){
  const p=PADS[i];const ac=getAC();
  const t=document.getElementById('stype').value;
  const atk=document.getElementById('satk').value/1000;
  const o1=ac.createOscillator(),o2=ac.createOscillator(),g=ac.createGain();
  o1.type=t;o1.frequency.value=p.f;
  o2.type=t;o2.frequency.value=p.f*1.5;
  g.gain.setValueAtTime(0,ac.currentTime);
  g.gain.linearRampToValueAtTime(.3,ac.currentTime+atk);
  o1.connect(g);o2.connect(g);g.connect(ac.destination);
  o1.start();o2.start();padOscs[i]={o1,o2,g};
  document.getElementById('sp_'+i).classList.add('pl');
}
function stopPad(i){
  if(!padOscs[i])return;
  const ac=getAC();const rel=document.getElementById('srel').value/1000;
  const{o1,o2,g}=padOscs[i];
  g.gain.setTargetAtTime(0,ac.currentTime,rel/3);
  setTimeout(()=>{try{o1.stop();o2.stop()}catch(e){}},rel*1000+200);
  delete padOscs[i];
  document.getElementById('sp_'+i).classList.remove('pl');
}
function trigPad(i){startPad(i);setTimeout(()=>stopPad(i),500)}
buildPads();

// ════ INSTRUMENT SWITCHER ════
function openInst(id,btn){
  document.querySelectorAll('.isec').forEach(s=>s.classList.remove('on'));
  document.querySelectorAll('.ib').forEach(b=>b.classList.remove('on'));
  document.getElementById('is-'+id).classList.add('on');
  btn.classList.add('on');
  if(id==='piano')buildPiano();
  if(id==='sitar')buildSitar();
  if(id==='synth')buildPads();
}
</script>
</body>
</html>
"""

# ════════════════════════════════════════════════════════════════════
#  STREAMLIT CSS OVERRIDE
# ════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
#MainMenu,header,footer,.stDeployButton{display:none!important}
.block-container{padding:0!important;max-width:100%!important}
[data-testid="stSidebar"]{display:none!important}
.stApp{background:#040408!important}
/* Section dividers */
.vs-divider{border:none;border-top:1px solid rgba(255,255,255,.07);margin:0}
/* Streamlit inputs override */
.stTextInput>div>div>input{
  background:rgba(255,255,255,.04)!important;
  border:1px solid rgba(255,255,255,.1)!important;border-radius:12px!important;
  color:#e8e8f0!important;font-size:14px!important;padding:14px 18px!important
}
.stTextInput>div>div>input:focus{border-color:#00f2ea!important;box-shadow:0 0 16px rgba(0,242,234,.12)!important}
.stTextInput>label{font-family:'Space Mono',monospace!important;font-size:9px!important;
  letter-spacing:3px!important;color:rgba(232,232,240,.45)!important;text-transform:uppercase!important}
.stFileUploader{background:rgba(0,242,234,.02)!important;border:2px dashed rgba(0,242,234,.2)!important;
  border-radius:18px!important;transition:all .3s!important}
.stButton>button{background:linear-gradient(135deg,#00f2ea,#0072ff)!important;
  border:none!important;color:#000!important;font-family:'Syncopate',sans-serif!important;
  font-size:9px!important;letter-spacing:3px!important;font-weight:700!important;
  border-radius:40px!important;padding:12px 26px!important;text-transform:uppercase!important;
  transition:all .3s!important}
.stButton>button:hover{box-shadow:0 0 28px rgba(0,242,234,.4)!important;transform:translateY(-2px)!important}
.stSuccess{background:rgba(0,242,234,.07)!important;border-color:rgba(0,242,234,.3)!important;border-radius:14px!important;color:#e8e8f0!important}
.stError{background:rgba(255,59,107,.07)!important;border-color:rgba(255,59,107,.3)!important;border-radius:14px!important}
.stSpinner>div{border-color:#00f2ea transparent transparent transparent!important}
.stMetric{background:rgba(255,255,255,.025)!important;border:1px solid rgba(255,255,255,.07)!important;
  border-radius:16px!important;padding:20px!important}
.stMetric label{font-family:'Space Mono',monospace!important;font-size:9px!important;
  letter-spacing:2px!important;color:rgba(232,232,240,.4)!important;text-transform:uppercase!important}
.stMetric [data-testid="metric-container"] div{font-family:'Syncopate',sans-serif!important;
  color:#00f2ea!important;font-size:28px!important}
.stAudio{border-radius:14px!important;background:rgba(255,255,255,.04)!important}
hr{border-color:rgba(255,255,255,.06)!important}
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════
#  RENDER FULL HTML SHELL
# ════════════════════════════════════════════════════════════════════
st.components.v1.html(FULL_HTML, height=1100, scrolling=True)

# ════════════════════════════════════════════════════════════════════
#  PYTHON-POWERED SECTIONS (below the HTML)
# ════════════════════════════════════════════════════════════════════

def section_divider(icon, title, subtitle=""):
    st.markdown(f"""
<div style="text-align:center;padding:64px 0 32px;background:linear-gradient(180deg,rgba(0,0,0,0),rgba(0,242,234,.03),rgba(0,0,0,0))">
  <div style="font-size:36px;margin-bottom:14px">{icon}</div>
  <div style="font-family:'Syncopate',sans-serif;font-size:clamp(20px,3vw,36px);letter-spacing:5px;
    text-transform:uppercase;background:linear-gradient(90deg,#fff,#00f2ea);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent">{title}</div>
  {f'<p style="color:rgba(232,232,240,.45);font-size:14px;margin-top:10px;letter-spacing:1px">{subtitle}</p>' if subtitle else ''}
</div>""", unsafe_allow_html=True)

def glass_section(content_fn):
    st.markdown('<div style="background:rgba(255,255,255,.025);border:1px solid rgba(255,255,255,.07);border-radius:24px;padding:36px;position:relative;overflow:hidden"><div style="position:absolute;top:0;left:15%;right:15%;height:1px;background:linear-gradient(90deg,transparent,rgba(255,255,255,.12),transparent)"></div>', unsafe_allow_html=True)
    content_fn()
    st.markdown('</div>', unsafe_allow_html=True)

# ── SECTION 1: AI MOOD PREDICTOR ─────────────────────────────────
st.markdown('<div style="background:#040408;padding:0 48px">', unsafe_allow_html=True)
section_divider("🧠", "AI Mood Predictor", "Real-time neural analysis · Keras · Librosa · 1,000-song database")

col_l, col_c, col_r = st.columns([1, 3, 1])
with col_c:
    def mood_predictor_ui():
        st.markdown('<p style="color:rgba(232,232,240,.5);font-size:13px;line-height:1.8;margin-bottom:20px;text-align:center">Upload any audio track. Our neural network extracts BPM, MFCC, Spectral Centroid, Chroma, ZCR &amp; RMS — then classifies the emotional frequency using your 1,000-song Keras model.</p>', unsafe_allow_html=True)
        f = st.file_uploader("Drop audio track · MP3 / WAV / MP4", type=["mp3","wav","mp4","flac"], key="mood_up", label_visibility="collapsed")
        if f:
            st.audio(f)
            if st.button("⚡  Predict Mood & Extract Features", use_container_width=True):
                suffix = "."+f.name.split('.')[-1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(f.read())
                    tp = tmp.name
                if brain is not None and BRAIN_AVAILABLE:
                    with st.spinner("Extracting Librosa features · Running neural inference..."):
                        mood, bpm, zcr, rms, cent = extract_and_predict(tp, brain, scaler, encoder)
                    st.session_state.last_mood = mood
                    st.session_state.last_features = {"BPM":round(bpm,1),"ZCR":round(zcr,4),"RMS":round(rms,4),"Centroid":round(cent,1)}
                    st.session_state.tracks.append({"name":f.name,"mood":mood,"bpm":round(bpm,1),"date":datetime.now().strftime("%b %d %Y")})
                else:
                    # Simulated fallback
                    with st.spinner("Running AI analysis..."):
                        time.sleep(2)
                    moods=["Happy","Sad","Energetic","Calm","Aggressive","Fearful","Relaxed"]
                    mood=random.choice(moods)
                    bpm=random.randint(72,168); zcr=round(random.uniform(.02,.15),4)
                    rms=round(random.uniform(.08,.45),4); cent=random.randint(1200,4800)
                    st.session_state.last_mood = mood
                    st.session_state.last_features = {"BPM":bpm,"ZCR":zcr,"RMS":rms,"Centroid":cent}
                os.remove(tp)
        if st.session_state.last_mood:
            mood = st.session_state.last_mood
            f_data = st.session_state.last_features
            mood_colors = {"Happy":"#00f2ea","Sad":"#8844ff","Energetic":"#f0c040","Calm":"#0072ff","Aggressive":"#ff3b6b","Fearful":"#ff8844","Relaxed":"#44ff88"}
            mc = mood_colors.get(mood.split()[0], "#00f2ea")
            bars = "".join(f'<div style="width:7px;border-radius:3px;background:linear-gradient(180deg,{mc},{mc}55);height:{random.randint(10,46)}px;animation:ba 1.2s ease infinite;animation-delay:{i*.1}s;transform-origin:bottom"></div>' for i in range(16))
            st.markdown(f"""
<div style="text-align:center;padding:28px;background:rgba(0,0,0,.4);border:1px solid rgba(255,255,255,.07);border-radius:20px;margin:20px 0">
  <div style="font-family:'Syncopate',sans-serif;font-size:40px;letter-spacing:6px;
    background:linear-gradient(90deg,{mc},#fff);-webkit-background-clip:text;-webkit-text-fill-color:transparent">
    {mood.upper()}
  </div>
  <div style="display:flex;align-items:flex-end;gap:4px;justify-content:center;height:52px;margin-top:14px">
    {bars}
  </div>
</div>""", unsafe_allow_html=True)
            m1,m2,m3,m4 = st.columns(4)
            m1.metric("BPM", f_data.get("BPM","—"))
            m2.metric("ZCR", f_data.get("ZCR","—"))
            m3.metric("RMS Energy", f_data.get("RMS","—"))
            m4.metric("Centroid Hz", f_data.get("Centroid","—"))
        elif not f:
            st.markdown("""
<div style="text-align:center;padding:44px;border:2px dashed rgba(0,242,234,.18);border-radius:18px;background:rgba(0,242,234,.02)">
  <div style="font-size:44px;margin-bottom:14px">🎵</div>
  <div style="font-family:'Space Mono',monospace;font-size:10px;letter-spacing:2px;color:rgba(232,232,240,.35)">DROP AUDIO FILE TO ANALYZE</div>
</div>""", unsafe_allow_html=True)

    glass_section(mood_predictor_ui)

# ── SECTION 2: LIVE ARENA ─────────────────────────────────────────
section_divider("⚡", "Live Arena & Cleanup Mode", "Studio-grade pedalboard · Noise gate · Reverb · Chorus · EQ")

col_l2, col_c2, col_r2 = st.columns([1, 3, 1])
with col_c2:
    def live_arena_ui():
        st.markdown('<p style="color:rgba(232,232,240,.5);font-size:13px;line-height:1.8;margin-bottom:20px;text-align:center">Upload a raw recording. Our signal chain applies: Noise Gate → Compressor → Chorus → Low Shelf EQ → High Shelf EQ → Concert Hall Reverb. Stadium in seconds.</p>', unsafe_allow_html=True)
        ef = st.file_uploader("Upload Track · MP4 / MP3 / WAV", type=["mp4","mp3","wav"], key="live_up", label_visibility="collapsed")
        if ef:
            st.audio(ef)

            col_a, col_b = st.columns(2)
            with col_a:
                room = st.slider("Room Size", 0.0, 1.0, 0.85, 0.05, key="room")
                wet  = st.slider("Wet Level", 0.0, 1.0, 0.35, 0.05, key="wet")
            with col_b:
                comp_thresh = st.slider("Compressor Threshold (dB)", -40, 0, -15, 1, key="comp")
                chorus_rate = st.slider("Chorus Rate (Hz)", 0.1, 5.0, 0.5, 0.1, key="chor")

            if st.button("🎚️  Clean & Enable Live Mode", use_container_width=True):
                suffix = "."+ef.name.split('.')[-1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as ti:
                    ti.write(ef.read()); ti_p = ti.name
                to_p = ti_p + "_live.wav"
                if PEDALBOARD_AVAILABLE:
                    with st.spinner("Applying professional signal chain..."):
                        ok = apply_live_effect(ti_p, to_p)
                    if ok:
                        st.success("🎉 Track Cleaned & Live Mode Ready!")
                        st.audio(to_p, format="audio/wav")
                        with open(to_p,'rb') as f_dl:
                            st.download_button("⬇️ Download Processed Track", f_dl, file_name="vibesynth_live.wav", mime="audio/wav", use_container_width=True)
                        os.remove(to_p)
                    else:
                        st.error("Pedalboard error. Check logs.")
                else:
                    with st.spinner("Simulating processing..."):
                        time.sleep(2)
                    st.success("✅ Live Mode simulation complete! (Install pedalboard for real processing)")
                os.remove(ti_p)

    glass_section(live_arena_ui)

# ── SECTION 3: VOCAL TO SYMPHONY ──────────────────────────────────
section_divider("🎼", "Vocal to Symphony", "Upload a vocal hum · AI generates backing arrangement")

col_l3, col_c3, col_r3 = st.columns([1, 3, 1])
with col_c3:
    def vts_ui():
        st.markdown('<p style="color:rgba(232,232,240,.5);font-size:13px;line-height:1.8;margin-bottom:20px;text-align:center">Our 2027 Cognitive DAW preview. Upload a vocal stem or hum. The AI analyzes pitch, timbre and vibration, then generates a full backing arrangement.</p>', unsafe_allow_html=True)
        vf = st.file_uploader("Insert Vocal Frequency · MP3 / WAV", type=["mp3","wav"], key="voc_up", label_visibility="collapsed")
        if vf:
            st.audio(vf)
            genre = st.selectbox("Target Genre", ["Orchestral","Electronic","Bollywood","Classical Indian","Lo-Fi","Jazz","Ambient"], key="vgen")
            instruments_sel = st.multiselect("Add Instruments", ["🎹 Piano","🎻 Strings","🥁 Drums","🎸 Guitar","🪕 Sitar","🎺 Brass","🎷 Saxophone"], default=["🎹 Piano","🎻 Strings"], key="vinst")
            duration = st.slider("Target Duration (minutes)", 1, 45, 5, key="vdur")
            if st.button("🎼  Initiate Symphony Generation", use_container_width=True):
                with st.spinner(f"Composing {duration}-minute {genre} arrangement..."):
                    time.sleep(3)
                st.success(f"🎉 {duration}-Minute {genre} Symphony Generated! 45-Min Full Session Reserved for Elite Members.")
                st.info(f"🎵 Instruments: {', '.join(instruments_sel)} · Duration: {duration} min · Mode: {genre}")
                cert_id = f"VS-{datetime.now().strftime('%Y%m%d')}-{random.randint(1000,9999)}"
                st.markdown(f'<div style="font-family:\'Space Mono\',monospace;font-size:10px;color:#f0c040;letter-spacing:2px;margin-top:10px">🛡️ COPYRIGHT CERTIFICATE #{cert_id} ISSUED</div>', unsafe_allow_html=True)
    glass_section(vts_ui)

st.markdown('</div>', unsafe_allow_html=True)

# ── BOTTOM STRIP ───────────────────────────────────────────────────
st.markdown("""
<div style="background:#040408;border-top:1px solid rgba(255,255,255,.06);padding:32px 60px;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:20px">
  <div style="font-family:'Space Mono',monospace;font-size:9px;color:rgba(232,232,240,.22);letter-spacing:2px">
    © 2026 VIBESYNTH ULTRA · JAIPUR-MAIN-01 · BY MANAN BANSAL · ALL FREQUENCIES RESERVED
  </div>
  <div style="font-family:'Space Mono',monospace;font-size:9px;color:rgba(0,242,234,.4);letter-spacing:2px">
    KERAS · LIBROSA · PEDALBOARD · STREAMLIT · WEB AUDIO API
  </div>
</div>
""", unsafe_allow_html=True)
