import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 1. Load and Clean (Keep your cleaning fix for BPM!)
df = pd.read_csv('music_database_1000.csv')
df['BPM'] = df['BPM'].apply(lambda x: float(str(x).replace('[', '').replace(']', '')))

# 2. Update Features (X now has 7 columns)
X = df[['BPM', 'MFCC', 'Centroid', 'Rolloff', 'Chroma', 'ZCR', 'RMS']].values
y = df['Label'].values

# 3. Encode Genres (Rock -> 0, Jazz -> 1, etc.)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 4. Split and Scale
# Fixed the 'test_size' typo here as well
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ... (rest of your model architecture code stays the same)

# 5. Updated Neural Network (input_shape=(7,))
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(7,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# 6. Train the Brain
print("🧠 Training the 1,000-song model...")
model.fit(X_train, y_train, epochs=100, batch_size=10, validation_data=(X_test, y_test))

# 7. Save for Web & Mobile
model.save('music_mood_model_1000.keras')
print("✅ Success! 'music_mood_model_1000.keras' is born.")