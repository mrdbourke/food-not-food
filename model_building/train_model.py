import os
import tensorflow as tf
import argparse
import logging

print(f"[INFO] TensorFlow version: {tf.__version__}")

assert tf.__version__.startswith("2")
# Filter out messages, see here: https://stackoverflow.com/a/38645250/7900723
tf.get_logger().setLevel(logging.ERROR)

from pathlib import Path
from datetime import date
from tflite_model_maker import image_classifier
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
parser.add_argument(
    "--use_data_aug", default=True, help="Whether to use data augmentation or not."
)
parser.add_argument(
    "--model_spec",
    default="efficientnet_lite0",
    help="Which efficientnet-liteX model to use, X can be 0, 1, 2, 3, 4",
)
args = parser.parse_args()

# Setup parser variables
train_data_path = args.train_dir
test_data_path = args.test_dir
export_dir = Path(args.export_dir)
NUM_EPOCHS = int(args.epochs)
MODEL_SPEC = args.model_spec


# Find class names
class_names = sorted(os.listdir(train_data_path))
print(f"[INFO] Found class names: {class_names}")

# Create data loader
train_data = DataLoader.from_folder(train_data_path)
test_data = DataLoader.from_folder(test_data_path)

# Create model
print(f"[INFO] Creating and training model...")
print(f"[INFO] Training {MODEL_SPEC} for {NUM_EPOCHS} epochs...")
model = image_classifier.create(
    train_data=train_data,
    model_spec=MODEL_SPEC,
    epochs=NUM_EPOCHS,
    use_augmentation=args.use_data_aug,
)

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
    export_dir,
    f"{current_date}_food_not_food_model_{MODEL_SPEC}_v{model_number}.tflite",
)

print(f"[INFO] Saving the model to '{export_dir}' directory as '{model_save_path}'...")
model.export(export_dir=export_dir, tflite_filename=model_save_path)
print(f"[INFO] Model saved to: '{model_save_path}'.")
