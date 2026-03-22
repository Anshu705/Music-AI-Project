import librosa
import numpy as np
import keras
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os
import shutil

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

model = keras.models.load_model('music_mood_model_104.h5')
data = pd.read_csv('music_database_104.csv')
X_raw = data[['BPM', 'MFCC_Mean', 'Brightness']].values
scaler = StandardScaler().fit(X_raw)
encoder = LabelEncoder()
encoder.fit(data['Mood'])

# --- DYNAMIC FOLDER SEARCH ---
base_path = r'C:\Music_AI_Project'
# This looks for ANY folder containing your 104 songs
source_folder = None
for folder in os.listdir(base_path):
    full_path = os.path.join(base_path, folder)
    if os.path.isdir(full_path) and "music_env" not in folder and "Sorted" not in folder:
        source_folder = full_path
        break

output_base = os.path.join(base_path, 'Sorted_Music')

def sort_songs():
    if not source_folder or not os.path.exists(source_folder):
        print(f"❌ Error: Could not find your music folder in {base_path}")
        return

    if not os.path.exists(output_base):
        os.makedirs(output_base)

    print(f"🚀 Sorting songs from: {source_folder}")

    for file in os.listdir(source_folder):
        if file.endswith('.mp3'):
            try:
                path = os.path.join(source_folder, file)
                y, sr = librosa.load(path, duration=15) # Shorter duration = Faster sorting
                tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
                
                val_tempo = tempo[0] if isinstance(tempo, np.ndarray) else tempo
                raw_features = np.array([[val_tempo, np.mean(mfcc), np.mean(centroid)]])
                scaled_features = scaler.transform(raw_features)
                
                prediction = model.predict(scaled_features, verbose=0)
                mood = encoder.inverse_transform([np.argmax(prediction)])[0]
                
                mood_folder = os.path.join(output_base, mood)
                if not os.path.exists(mood_folder):
                    os.makedirs(mood_folder)
                
                shutil.copy(path, os.path.join(mood_folder, file))
                print(f"✅ AI labeled '{file}' as {mood}")
            except Exception as e:
                print(f"❌ Error sorting {file}: {e}")

sort_songs()