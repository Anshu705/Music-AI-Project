import os
import librosa
import pandas as pd
import numpy as np

base_path = 'C:/Music_AI_Project/archive/Data/genres_original'
genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
all_data = []

print("🚀 Starting Advanced Extraction (7 Features) for 1,000 songs...")

for genre in genres:
    folder_path = os.path.join(base_path, genre)
    for filename in os.listdir(folder_path):
        if filename.endswith('.wav'):
            file_path = os.path.join(folder_path, filename)
            try:
                y, sr = librosa.load(file_path, duration=30)
                
                # --- ADVANCED FEATURE SET ---
                tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
                mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13))
                spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
                spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
                chroma_stft = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))
                zcr = np.mean(librosa.feature.zero_crossing_rate(y))
                rms = np.mean(librosa.feature.rms(y=y)) # Volume/Energy
                
                all_data.append([filename, tempo, mfcc, spectral_centroid, spectral_rolloff, chroma_stft, zcr, rms, genre])
            except:
                print(f"⚠️ Skipping: {filename}")

cols = ['Filename', 'BPM', 'MFCC', 'Centroid', 'Rolloff', 'Chroma', 'ZCR', 'RMS', 'Label']
df = pd.DataFrame(all_data, columns=cols)
df.to_csv('music_database_1000.csv', index=False)
print("✅ Done! 1,000 songs analyzed with high-precision features.")