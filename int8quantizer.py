
import tensorflow as tf
import numpy as np
import os
import cv2

# Settings
IMG_SIZE = 512

# Configure paths (update these for your environment)
REPRESENTATIVE_IMAGES_DIR = "/path/to/representative/images"
TFLITE_INT8_MODEL = "/path/to/output/best_int8.tflite"
SAVED_MODEL_DIR = "/path/to/saved_model"

# Ensure the output directory exists
os.makedirs(os.path.dirname(TFLITE_INT8_MODEL), exist_ok=True)

# Representative dataset generator (limits to 100 images)
def representative_dataset():
    image_files = sorted([
        os.path.join(REPRESENTATIVE_IMAGES_DIR, f)
        for f in os.listdir(REPRESENTATIVE_IMAGES_DIR)
        if f.endswith(".jpg")
    ])[:100]
    for img_path in image_files:
        img = cv2.imread(img_path)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        yield [img]

# Convert the SavedModel to a TFLite INT8 model with float32 I/O
print(f"Loading SavedModel from: {SAVED_MODEL_DIR}")
converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_DIR)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.inference_input_type = tf.float32
converter.inference_output_type = tf.float32

print("Converting to INT8 with float32 I/O...")
tflite_quant_model = converter.convert()

# Save the quantized TFLite model
with open(TFLITE_INT8_MODEL, "wb") as f:
    f.write(tflite_quant_model)

print(f"Quantized model saved to: {TFLITE_INT8_MODEL}")
