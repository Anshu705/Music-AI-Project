import librosa
import numpy as np
import keras
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# 1. Load the Pro Brain
model = keras.models.load_model('music_mood_model_104.h5')

# 2. Setup the Math (Must match your 104-song table)
data = pd.read_csv('music_database_104.csv')
X_raw = data[['BPM', 'MFCC_Mean', 'Brightness']].values
scaler = StandardScaler().fit(X_raw)

encoder = LabelEncoder()
encoder.fit(data['Mood'])

def predict_mood(file_path):
    y, sr = librosa.load(file_path, duration=30)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    
    val_tempo = tempo[0] if isinstance(tempo, np.ndarray) else tempo
    raw_features = np.array([[val_tempo, np.mean(mfcc), np.mean(centroid)]])
    
    # Apply the same scaling used in training
    scaled_features = scaler.transform(raw_features)
    
    prediction = model.predict(scaled_features, verbose=0)
    return encoder.inverse_transform([np.argmax(prediction)])[0]

# --- THE TEST ---
test_song = r'C:\Music_AI_Project\test_song.mp3'

if os.path.exists(test_song):
    result = predict_mood(test_song)
    print("\n" + "="*35)
    print(f"🎵 TESTING: {os.path.basename(test_song)}")
    print(f"🔥 AI PREDICTION: {result}")
    print("="*35 + "\n")
else:
    print(f"❌ Error: {test_song} not found!")