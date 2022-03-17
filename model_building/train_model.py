import os

import numpy as np

import tensorflow as tf

import matplotlib.pyplot as plt
import argparse

print(tf.__version__)

assert tf.__version__.startswith("2")
tf.get_logger().setLevel("ERROR")

from pathlib import Path
from datetime import date
from tflite_model_maker import model_spec
from tflite_model_maker import image_classifier
from tflite_model_maker.config import ExportFormat
from tflite_model_maker.config import QuantizationConfig
from tflite_model_maker.image_classifier import DataLoader

# Setup argparser
parser = argparse.ArgumentParser()
parser.add_argument(
    "--train_dir", default="../data/train", help="Training data directory."
)
parser.add_argument(
    "--test_dir", default="../data/test", help="Testing data directory."
)
parser.add_argument(
    "--export_dir", default="../models/", help="Directory to save trained models."
)
parser.add_argument(
    "--epochs", default=5, help="Number of epochs to train the model for."
)
args = parser.parse_args()

# Setup parser variables
train_data_path = args.train_dir
test_data_path = args.test_dir
export_dir = Path(args.export_dir)
NUM_EPOCHS = args.epochs


# Find class names
class_names = sorted(os.listdir(train_data_path))
print(f"[INFO] Found class names: {class_names}")

# Create data loader
train_data = DataLoader.from_folder(train_data_path)
test_data = DataLoader.from_folder(test_data_path)

# Create model
print(f"[INFO] Creating and training model...")
model = image_classifier.create(train_data, epochs=NUM_EPOCHS)

# Evaluate model
print(f"[INFO] Evaluating the model on test data...")
test_loss, test_accuracy = model.evaluate(test_data)
print(f"[INFO] Test loss: {test_loss:.4f} | Test accuracy: {test_accuracy*100:.3f}%")

# Save the model
def get_model_number(export_dir):
    num_models = len(os.listdir(export_dir))
    return num_models + 1


model_number = get_model_number(export_dir)

# Get current date
current_date = str(date.today())
model_save_path = Path(
    export_dir, f"{current_date}_food_not_food_model_v{model_number}.tflite"
)

print(f"[INFO] Saving the model to '{export_dir}' directory as '{model_save_path}'...")
model.export(export_dir=export_dir, tflite_filename=model_save_path)
print(f"[INFO] Model saved.")
