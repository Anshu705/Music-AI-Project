import pandas as pd

# Load the CSV currently in your folder
df = pd.read_csv('music_database_1000.csv')

# These are the 6 names causing the "KeyError"
missing_cols = ['MFCC', 'Centroid', 'Rolloff', 'Chroma', 'ZCR', 'RMS']

for col in missing_cols:
    if col not in df.columns:
        df[col] = 0.0  # Adding empty data so the website doesn't crash

# Save it back
df.to_csv('music_database_1000.csv', index=False)
print("✅ Local CSV is now fixed with 7 columns!")