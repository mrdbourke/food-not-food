"""
Downloads a random subset of images from Open Images.

Uses FiftyOne: https://voxel51.com/docs/fiftyone/tutorials/open_images.html
"""
import fiftyone as fo
import fiftyone.zoo as foz
import os

targ_dir = "data/open_images"

if os.path.exists(targ_dir):
    pass
else:
    os.makedirs(targ_dir)

dataset = foz.load_zoo_dataset(
    "open-images-v6",
    split=None,  # use split=None to download from all 3 splits (train, val, test)
    label_types=["classifications"],  # only get classification level labels
    max_samples=100,
    seed=42,  # set seed to download same images each time
    shuffle=True,
    dataset_dir=targ_dir,
)

session = fo.launch_app(dataset)
