import json
import numpy as np
import tensorflow as tf
from PIL import Image


MODEL_PATH = "models/clothing_mobilenetv2_final.keras"
CLASS_NAMES_PATH = "config/class_names.json"

IMAGE_SIZE = (180, 180)


# --------------------------------------------------
# Load model
# --------------------------------------------------

def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


# --------------------------------------------------
# Load class names
# --------------------------------------------------

def load_class_names():
    with open(CLASS_NAMES_PATH, "r") as f:
        return json.load(f)


# --------------------------------------------------
# Preprocess image
# --------------------------------------------------

def preprocess_image(image, image_size=(180, 180)):
    """
    Preprocess image for model inference.
    """

    image = image.convert("RGB")

    image = image.resize(image_size)

    image_array = np.asarray(
        image,
        dtype=np.float32
    )

    image_array /= 255.0

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# --------------------------------------------------
# Predict
# --------------------------------------------------

def predict_image(model, image, class_names, top_k=3):

    input_shape = model.input_shape

    image_height = input_shape[1]
    image_width = input_shape[2]

    processed_image = preprocess_image(
        image,
        image_size=(image_width, image_height)
    )

    predictions = model.predict(
        processed_image,
        verbose=0
    )[0]

    top_indices = np.argsort(
        predictions
    )[::-1][:top_k]

    results = []

    for index in top_indices:

        results.append({
            "class": class_names[index],
            "confidence": float(predictions[index])
        })

    return results