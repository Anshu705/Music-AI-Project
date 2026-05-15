import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

print("🧠 Booting up VibeSynth AI Training Module...")

# 1. Load and Clean
df = pd.read_csv('music_database_1000.csv')
# Keep your cleaning fix for BPM just in case
df['BPM'] = df['BPM'].apply(lambda x: float(str(x).replace('[', '').replace(']', '')))

print(f"📊 Loaded {len(df)} tracks. Preparing data...")

# 2. Update Features (X now has 7 columns)
X = df[['BPM', 'MFCC', 'Centroid', 'Rolloff', 'Chroma', 'ZCR', 'RMS']].values
y = df['Label'].values

# 3. Encode Moods (Happy -> 0, Sad -> 1, etc.)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Count exactly how many unique moods we have (should be 7)
num_classes = len(label_encoder.classes_)
print(f"🎯 Detected {num_classes} unique mood classes: {label_encoder.classes_}")

# 4. Scale the Entire Dataset
# We do this here so the scaler perfectly matches the one used in app.py
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Split the Scaled Data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# 6. Updated Neural Network Architecture
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(7,)),
    # Added Dropout layers to prevent the AI from just memorizing the songs (Overfitting)
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation='relu'),
    # Dynamically set the output layer to match the number of moods
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# 7. Train the Brain
print(f"🔥 Training the {len(df)}-song model...")
history = model.fit(
    X_train, y_train, 
    epochs=100, 
    batch_size=10, 
    validation_data=(X_test, y_test),
    verbose=1 # Change to 0 if you don't want to see the 100 epochs scroll by
)

# 8. Evaluate and Save
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"🎯 Final AI Accuracy on unseen data: {accuracy * 100:.2f}%")

model.save('music_mood_model_1000.keras')
print("✅ Success! 'music_mood_model_1000.keras' is born and ready for VibeSynth.")