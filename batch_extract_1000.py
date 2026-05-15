import os
import librosa
import pandas as pd
import numpy as np

# Path to your extracted GTZAN folders
base_path = './archive/Data/genres_original'
genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

# The essential mapping to convert genres into your 7 mood classes
mood_mapping = {
    'pop': 'Happy',
    'disco': 'Happy',
    'blues': 'Sad',
    'classical': 'Sad',
    'hiphop': 'Energetic',
    'metal': 'Aggressive',
    'rock': 'Aggressive',
    'reggae': 'Relaxed',
    'country': 'Relaxed',
    'jazz': 'Calm'
}

all_data = []
print("🚀 Upgrading Database to 7 Features and Mapping Moods...")

for genre in genres:
    folder_path = os.path.join(base_path, genre)
    if not os.path.exists(folder_path):
        print(f"⚠️ Folder not found: {genre}")
        continue
    
    # Get the correct mood label for this folder
    mood_label = mood_mapping.get(genre, 'Unknown')
    print(f"Processing {genre}... (Mapping to: {mood_label})")
    
    for filename in os.listdir(folder_path):
        if filename.endswith('.wav'):
            file_path = os.path.join(folder_path, filename)
            try:
                # Load the audio file
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
                
                # IMPORTANT: Append the mood_label, NOT the genre!
                all_data.append([filename, bpm, mfcc, centroid, rolloff, chroma, zcr, rms, mood_label])
            except Exception as e:
                print(f"❌ Error processing {filename}: {e}")

# Save the final version
df = pd.DataFrame(all_data, columns=['Filename', 'BPM', 'MFCC', 'Centroid', 'Rolloff', 'Chroma', 'ZCR', 'RMS', 'Label'])
df.to_csv('music_database_1000.csv', index=False)
print("✨ Complete! Database is now mapped to Moods. You can retrain your model or push to GitHub.")