"""
Evaluate a model on a target image.
"""
import argparse


import os
import re

import numpy as np

import tensorflow as tf

assert tf.__version__.startswith("2")
tf.get_logger().setLevel("INFO")

from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_path",
    default="../models/food_not_food_model_v5.tflite",
    help="The path to the target model to test.",
)
parser.add_argument(
    "--image_path",
    default="../test_food_not_food_images/chicken_wings.jpeg",
    help="The path to the target image to make a prediction with the model on.",
)
args = parser.parse_args()

# Get paths
model_path = str(args.model_path)
image_path = str(args.image_path)

# Load model
print(f"[INFO] Loading model from '{model_path}'...")
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]

print(input_index, output_index)

# Load and pred on image
def load_image(image_path, img_shape=224):
    print(f"[INFO] Loading image from {image_path}...")
    # Read in the image
    img = tf.io.read_file(image_path)
    # Decode it into a tensor
    img = tf.image.decode_jpeg(img)
    # Resize the image
    img = tf.image.resize(img, [img_shape, img_shape])

    # Turn img to uint8 (for TFLite)
    img = tf.cast(img, dtype=tf.uint8)

    # Expand dimensions for batch size
    img = tf.expand_dims(img, axis=0)

    return img


img = load_image(image_path=image_path)

# Make prediction - source: https://stackoverflow.com/a/68132736/7900723
print(f"[INFO] Making prediction on image...")
interpreter.set_tensor(input_index, img)
interpreter.invoke()
output = interpreter.get_tensor(output_index)
class_names = {0: "food", 1: "not_food"}
print(f"[INFO] Model prediction: {class_names[output.argmax()]}")
print(f"[INFO] Model outputs: {output}")

