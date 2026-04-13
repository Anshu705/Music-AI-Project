import tensorflow as tf
import numpy as np

# 1. Load the model
model = tf.keras.models.load_model('music_mood_model_104.keras')

# 2. Define the input shape (3 features: BPM, MFCC, Brightness)
# We use a batch size of 1 for mobile inference
run_model = tf.function(lambda x: model(x))
concrete_func = run_model.get_concrete_function(
    tf.TensorSpec([1, 3], model.inputs[0].dtype)
)

# 3. Convert using the concrete function
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func], model)
tflite_model = converter.convert()

# 4. Save the file
with open('music_mood_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("✅ Success! 'music_mood_model.tflite' is now ready for your OnePlus Nord CE4.")