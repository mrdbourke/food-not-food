"""
Extracts images from Food101 into "food" image category.

No sorting is required since all images are of food.

Training, evaluation and test sets are ignored.

Original data downloaded here: https://www.kaggle.com/dansbecker/food-101
"""
import random
import pathlib
import argparse
import os

from shutil import copy2

parser = argparse.ArgumentParser()
parser.add_argument(
    "-t",
    "--targ_dir",
    default="../data/101_food_classes_all_data",
    help="target directory where downloaded Food101 images are",
)
parser.add_argument(
    "-d",
    "--dest_dir",
    default="../data/101_food_classes_all_data_extracted",
    help="destination directory to extract images from targ_dir to",
)
parser.add_argument("-s", "--seed", default=42, help="random seed for selecting images")
parser.add_argument(
    "-n",
    "--num_samples",
    default=100,
    help="number of images to randomly sample from Food101",
)

args = parser.parse_args()
targ_dir = args.targ_dir
dest_dir = args.dest_dir
NUM_SAMPLES = int(args.num_samples)

assert os.path.exists(targ_dir), "Target directory does not exist, please create it"

# Get a list of paths from targ_dir
# Note: Images are stored in form "101_food_classes_all_data/train/apple_pie/image.jpg"
image_path_list = list(pathlib.Path(targ_dir).glob("*/*/*.jpg"))

# Randomly sample image paths from Food101 data
random.seed(args.seed)
print(f"[INFO] Sampling {NUM_SAMPLES} images from {targ_dir}...")
random_image_path_sample = random.sample(image_path_list, k=NUM_SAMPLES)

# Copy randomly sampled images from targ_dir to dest_dir
for image_path in random_image_path_sample:
    destination_folder = pathlib.Path(dest_dir).joinpath("food_images")
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder, exist_ok=True)
    destination_path = pathlib.Path(destination_folder).joinpath(image_path.name)
    print(f"[INFO] Copying {image_path} to {destination_path}...")
    copy2(image_path, destination_path)

# Print out images extracted
for dir, subdirs, files in os.walk(args.dest_dir):
    for subdir in subdirs:
        subdir_path = os.path.join(dir, subdir)
        print(
            f"[INFO] Total {subdir} images in {subdir_path}: {len(os.listdir(subdir_path))}"
        )
