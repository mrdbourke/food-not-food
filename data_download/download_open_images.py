"""
Downloads a random subset of images from Open Images.

Uses FiftyOne: https://voxel51.com/docs/fiftyone/tutorials/open_images.html
"""
import fiftyone as fo
import fiftyone.zoo as foz
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "-t",
    "--targ_dir",
    default="../data/open_images",
    help="target directory to download images to",
)
parser.add_argument(
    "-s", "--seed", default=42, help="random seed for downloading images"
)
parser.add_argument(
    "-n",
    "--num_samples",
    default=100,
    help="number of images to download from Open Images (downloads occur from \
        train, val, test sets by default so the actual number will be 3x this,\
            so 100 = 3x100 = 300 images downloaded",
)

args = parser.parse_args()

targ_dir = args.targ_dir
num_samples = int(args.num_samples)
seed = args.seed

# Print what's happening
print(f"Downloading {num_samples} images to {targ_dir} with seed: {seed}...")

# Create Target Directly
if not os.path.exists(targ_dir):
    os.makedirs(targ_dir)

# Download images from Open Images
dataset = foz.load_zoo_dataset(
    "open-images-v6",
    split=None,  # use split=None to download from all 3 splits (train, val, test)
    label_types=["classifications"],  # only get classification level labels
    max_samples=num_samples,
    seed=seed,  # set seed to download same images each time
    shuffle=True,
    dataset_dir=targ_dir,
)

# Launch an iteractive data session with Voxel51's data explorer
# session = fo.launch_app(dataset)
