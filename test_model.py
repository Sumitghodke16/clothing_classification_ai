import tensorflow as tf
import json

MODEL_PATH = "models/clothing_mobilenetv2_final.keras"
CLASS_NAMES_PATH = "config/class_names.json"

print("Loading model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")
print("Input shape:", model.input_shape)
print("Output shape:", model.output_shape)

with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)

print("\nNumber of classes:", len(class_names))

for i, name in enumerate(class_names):
    print(i, "->", name)