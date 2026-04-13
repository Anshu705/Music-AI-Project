import os
import librosa
import pandas as pd
import numpy as np

# Path to your extracted GTZAN folders
base_path = 'C:/Music_AI_Project/genres_original'
genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

all_data = []
print("🚀 Upgrading Database to 7 Features...")

for genre in genres:
    folder_path = os.path.join(base_path, genre)
    if not os.path.exists(folder_path): continue
    
    for filename in os.listdir(folder_path):
        if filename.endswith('.wav'):
            file_path = os.path.join(folder_path, filename)
            try:
                y, sr = librosa.load(file_path, duration=30)
                
                # 1. BPM
                tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
                bpm = float(tempo[0]) if isinstance(tempo, (list, np.ndarray)) else float(tempo)
                
                # 2-7. The 6 Missing Features
                mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13))
                centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
                rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
                chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))
                zcr = np.mean(librosa.feature.zero_crossing_rate(y))
                rms = np.mean(librosa.feature.rms(y=y))
                
                all_data.append([filename, bpm, mfcc, centroid, rolloff, chroma, zcr, rms, genre])
                print(f"✅ Extracted 7 features for: {filename}")
            except:
                continue

# Save the complete 7-feature database
df = pd.DataFrame(all_data, columns=['Filename', 'BPM', 'MFCC', 'Centroid', 'Rolloff', 'Chroma', 'ZCR', 'RMS', 'Label'])
df.to_csv('music_database_1000.csv', index=False)
print("✨ Database matches Model Features! Now push this to GitHub.")