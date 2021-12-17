"""
Splits data in data/food & data/not_food into:

data/
    train/
        food/
            ...
        not_food/
            ...
    test/
        food/
            ...
        not_food/
            ...
"""
import pathlib
import random
import os

from shutil import copy2, move

# Get all food image paths
targ_dir = "../data/"
food_dir = "../data/food"
not_food_dir = "../data/not_food"
food_dir_train, food_dir_test = os.path.join(targ_dir, "train", "food"), os.path.join(
    targ_dir, "test", "food"
)
not_food_dir_train, not_food_dir_test = os.path.join(
    targ_dir, "train", "not_food"
), os.path.join(targ_dir, "test", "not_food")


food_image_filepaths = list(pathlib.Path(food_dir).glob("*.jpg"))
not_food_image_filepaths = list(pathlib.Path(not_food_dir).glob("*.jpg"))

assert (
    len(food_image_filepaths) > 0
), "No food image filepaths found, check the directories"
assert (
    len(not_food_image_filepaths) > 0
), "No not food image filepaths found, check the directories"


# Shuffle list and get train and test splits
def create_train_test_split(image_filepath_list, seed=42):
    random.seed(seed)
    train_split = int(0.8 * len(image_filepath_list))
    print(train_split)
    train_image_split = random.sample(image_filepath_list, train_split)
    print(len(train_image_split))
    test_image_split = list(set(image_filepath_list).difference(set(train_image_split)))
    return train_image_split, test_image_split


def copy_images_to_file(image_filepath_list, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)
    for image_path in image_filepath_list:
        image_filename = image_path.name
        dest_path = pathlib.Path(target_dir).joinpath(image_filename)
        print(f"[INFO] Moving {image_path} to {dest_path}...")
        move(image_path, dest_path)


# Get food image paths (train and test)
food_train_image_list, food_test_image_list = create_train_test_split(
    food_image_filepaths
)

# Copy food image paths (train and test) to file
copy_images_to_file(
    image_filepath_list=food_train_image_list, target_dir=food_dir_train
)
copy_images_to_file(image_filepath_list=food_test_image_list, target_dir=food_dir_test)

# Get not food image paths (train and test)
not_food_train_image_list, not_food_test_image_list = create_train_test_split(
    not_food_image_filepaths
)

# Copy not food image paths (train and test) to file
copy_images_to_file(
    image_filepath_list=not_food_train_image_list, target_dir=not_food_dir_train
)
copy_images_to_file(
    image_filepath_list=not_food_test_image_list, target_dir=not_food_dir_test
)
