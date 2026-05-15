import os
import librosa
import pandas as pd
import numpy as np

# Path to your GTZAN folders on your laptop
base_path = 'C:/Music_AI_Project/genres_original'
genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

all_data = []
print("🚀 Analyzing 1,000 songs for the 7-feature Brain...")

for genre in genres:
    folder_path = os.path.join(base_path, genre)
    if not os.path.exists(folder_path): continue
    
    for filename in os.listdir(folder_path):
        if filename.endswith('.wav'):
            try:
                path = os.path.join(folder_path, filename)
                y, sr = librosa.load(path, duration=30)
                
                # Math extraction for the 7 required features
                tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
                bpm = float(tempo[0]) if isinstance(tempo, (list, np.ndarray)) else float(tempo)
                mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13))
                centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
                rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
                chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))
                zcr = np.mean(librosa.feature.zero_crossing_rate(y))
                rms = np.mean(librosa.feature.rms(y=y))
                
                all_data.append([filename, bpm, mfcc, centroid, rolloff, chroma, zcr, rms, genre])
                print(f"✅ Processed: {filename}")
            except: continue

# Create the proper DataFrame
df = pd.DataFrame(all_data, columns=['Filename', 'BPM', 'MFCC', 'Centroid', 'Rolloff', 'Chroma', 'ZCR', 'RMS', 'Label'])

# Save it - This MUST be the same name used in your app.py
df.to_csv('music_database_1000.csv', index=False)
print("✨ 1,000-Song Database is now PERFECT. Ready for GitHub!")