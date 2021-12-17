"""
Extracts images from Open Images (downloaded with download_open_images.py) into "food" and "not_food" categories,
training, evaluation and test sets are ignored.

Original data downloaded here: https://voxel51.com/docs/fiftyone/tutorials/open_images.html
"""
import pandas as pd
import os
import argparse
import pathlib
import random

from shutil import copy2

parser = argparse.ArgumentParser()
parser.add_argument(
    "-t",
    "--targ_dir",
    default="../data/open_images",
    help="target directory where downloaded images are",
)
parser.add_argument(
    "-d",
    "--dest_dir",
    default="../data/open_images_extracted",
    help="destination directory to extract images from targ_dir to",
)

args = parser.parse_args()
targ_dir = args.targ_dir

assert os.path.exists(targ_dir), "Target directory does not exist, please create it"

# Read in class names and turn to list and create classes and label dict
print(f"[INFO] Getting class names of Open Images data...")
targ_classes_path = os.path.join(targ_dir, "train", "metadata", "classes.csv")
classed_df = classes_df = pd.read_csv(targ_classes_path, names=["id", "class"])
class_list = classes_df["class"].tolist()
class_label_dict = dict(zip(classes_df["id"], classes_df["class"]))

# Read in food names list (from NLTK)
print(f"[INFO] Getting list of foods...")
with open("../data/food_list.txt", "r") as f:
    food_list = f.read().splitlines()

# Filter Open Images class list for food classes
print(f"[INFO] Filtering list of Open Images classes to find food classes...")
open_images_food_list = []
for class_name in class_list:
    if class_name.lower() in food_list:
        open_images_food_list.append(class_name)
print(
    f"[INFO] Found some foods, here's 10 random picks:"
    f"\t{random.sample(open_images_food_list, 10)}"
)

# Add column to classes_df saying whether the class is food or not
classes_df["is_food"] = classes_df["class"].isin(open_images_food_list)

# Get all image paths from targ_dir (this directory contains all downloaded Open Images)
image_path_list = list(pathlib.Path(targ_dir).glob("*/*/*.jpg"))

# Turn all image paths into a list of their IDs
image_ids = [image_path.name.split(".")[0] for image_path in image_path_list]

# Read in all label files from Open Images (train, val, test)
labels_list = list(pathlib.Path(targ_dir).glob("*/labels/classifications.csv"))

# Turn all labels into a single DataFrame (so it can be manipulated)
print(
    f"[INFO] Importing Open Images label files (train, val, test) and combining them..."
)
labels_df_list = []
for labels in labels_list:
    df = pd.read_csv(labels)
    labels_df_list.append(df)

labels_df = (pd.concat(labels_df_list, axis=0, ignore_index=True)).drop(
    ["Source", "Confidence"], axis=1
)

# Find out whether the image is the labels dataframe is downloaded downloaded
labels_df["downloaded"] = labels_df["ImageID"].isin(image_ids)

# Get a slice of the labels df of only image IDs that are downloaded
downloaded_labels = labels_df[labels_df["downloaded"] == True].copy()

# Add extra columns to the downloaded labels dataframe
downloaded_labels["class_name"] = downloaded_labels["LabelName"].map(class_label_dict)
downloaded_labels["is_food"] = downloaded_labels["class_name"].isin(
    open_images_food_list
)

# Create a function to extract food and not food image paths from dataframe
def get_food_and_not_food_image_path_lists(dataframe):
    food_image_path_list = []
    not_food_image_path_list = []
    for i, row in enumerate(dataframe.itertuples(index=False)):
        # Get image details
        image_id = row[0]
        class_name = row[3]
        is_food = row[4]

        # Get image path
        image_path = list(pathlib.Path(targ_dir).glob("*/*/" + image_id + ".jpg"))[0]

        # See if image is food or not
        if is_food:
            food_image_path_list.append(image_path)
        else:
            not_food_image_path_list.append(image_path)

    print(f"Found {len(food_image_path_list)} food image label paths")
    print(f"Found {len(not_food_image_path_list)} not food image label paths")
    print(f"Removing duplicates and food image paths from not food paths...")
    food_image_path_set = set(food_image_path_list)
    # Remove food image paths from not food paths (give food label priority)
    not_food_image_path_set = set(not_food_image_path_list) - food_image_path_set
    print(
        f"Updated lengths:"
        f"\n-> Food images: {len(food_image_path_set)}"
        f"\n-> Not food images: {len(not_food_image_path_set)}"
        f"\n-> Total images: {len(food_image_path_set) + len(not_food_image_path_set)}"
    )

    return list(food_image_path_set), list(not_food_image_path_set)


# Get food and not food image lists
# Note: these lists are mutually exclusive,
# this means you won't find overlapping images
# (food labels take priority)
print(f"[INFO] Filtering Open Images filepaths into food and not_food...")
food_image_path_list, not_food_image_path_list = get_food_and_not_food_image_path_lists(
    dataframe=downloaded_labels
)

# Create function to move image from downloaded Open Images to seperate food and not_food classes
def copy_image_to_folder(dir_name, image_path, is_food):
    if is_food:
        targ_class = "food_images"
        destination_folder = pathlib.Path(dir_name).joinpath(targ_class)
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder, exist_ok=True)
        destination_path = pathlib.Path(destination_folder).joinpath(image_path.name)
        print(f"Copying {image_path} to {destination_path}...")
        copy2(image_path, destination_path)
    else:
        targ_class = "not_food_images"
        destination_folder = pathlib.Path(dir_name).joinpath(targ_class)
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder, exist_ok=True)
        destination_path = pathlib.Path(destination_folder).joinpath(image_path.name)
        print(f"[INFO] Copying {image_path} to {destination_path}...")
        copy2(image_path, destination_path)


# Create function to copy list of images to targest destination
def copy_image_list_to_folder(dir_name, image_path_list, is_food):
    for i, image_path in enumerate(image_path_list):
        i += 1
        copy_image_to_folder(dir_name=dir_name, image_path=image_path, is_food=is_food)
    print(f"[INFO] Total {'food' if is_food else 'not_food'} images copied: {i}")


# Extract downloaded Open Images to food and not_food folders
print(f"[INFO] Copying images to {args.dest_dir}...")
copy_image_list_to_folder(
    dir_name=args.dest_dir, image_path_list=food_image_path_list, is_food=True
)

copy_image_list_to_folder(
    dir_name=args.dest_dir, image_path_list=not_food_image_path_list, is_food=False
)

# Print out images extracted
for dir, subdirs, files in os.walk(args.dest_dir):
    for subdir in subdirs:
        subdir_path = os.path.join(dir, subdir)
        print(
            f"[INFO] Total {subdir} images in {subdir_path}: {len(os.listdir(subdir_path))}"
        )
