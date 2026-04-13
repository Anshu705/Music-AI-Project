import os
import librosa
import pandas as pd
import numpy as np

# Path to your extracted GTZAN folders on Predator Helios
base_path = 'C:/Music_AI_Project/genres_original'
genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

all_data = []
print("🚀 Upgrading Database to 7 Features...")

for genre in genres:
    folder_path = os.path.join(base_path, genre)
    if not os.path.exists(folder_path):
        print(f"⚠️ Folder not found: {genre}")
        continue
    
    for filename in os.listdir(folder_path):
        if filename.endswith('.wav'):
            file_path = os.path.join(folder_path, filename)
            try:
                y, sr = librosa.load(file_path, duration=30)
                
                # Extracting the 7 features required by the new app.py
                tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
                bpm = float(tempo[0]) if isinstance(tempo, (list, np.ndarray)) else float(tempo)
                
                mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13))
                centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
                rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
                chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))
                zcr = np.mean(librosa.feature.zero_crossing_rate(y))
                rms = np.mean(librosa.feature.rms(y=y))
                
                all_data.append([filename, bpm, mfcc, centroid, rolloff, chroma, zcr, rms, genre])
            except Exception as e:
                print(f"❌ Error processing {filename}: {e}")

# Save the final version
df = pd.DataFrame(all_data, columns=['Filename', 'BPM', 'MFCC', 'Centroid', 'Rolloff', 'Chroma', 'ZCR', 'RMS', 'Label'])
df.to_csv('music_database_1000.csv', index=False)
print("✨ Complete! Push 'music_database_1000.csv' to GitHub now.")