import librosa
import pandas as pd
import numpy as np
import os

def process_single_folder(folder_path):
    data_list = []
    print(f"Reading songs from: {folder_path}")
    
    for file in os.listdir(folder_path):
        if file.endswith('.mp3'):
            try:
                path = os.path.join(folder_path, file)
                y, sr = librosa.load(path, duration=30)
                
                tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
                
                data_list.append({
                    'Song_Name': file,
                    'BPM': tempo[0] if isinstance(tempo, np.ndarray) else tempo,
                    'MFCC_Mean': np.mean(mfcc),
                    'Brightness': np.mean(centroid),
                    'Mood': 'TBD'  # You will fill this in the CSV
                })
                print(f"✅ Extracted: {file}")
            except Exception as e:
                print(f"❌ Skip {file}: {e}")
            
    df = pd.DataFrame(data_list)
    df.to_csv('music_database_104.csv', index=False)
    print(f"\n🚀 Done! Open 'music_database_104.csv' to add moods.")

# UPDATE THIS TO YOUR ACTUAL FOLDER NAME
process_single_folder(r'C:\Music_AI_Project\Mysong')