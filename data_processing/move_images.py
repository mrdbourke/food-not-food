"""
Moves images from extracted image files to ../data/food and ../data/not_food
"""
import os
import shutil
import pathlib


def move_images(source, destination):
    image_file_path_list = pathlib.Path(source).glob("*.jpg")
    for image_path in image_file_path_list:
        print(f"[INFO] Moving {image_path} to {destination}")
        shutil.move(src=os.path.join(image_path), dst=destination)


targ_dir = "../data"
food_image_dir = os.path.join(targ_dir, "food")
not_food_image_dir = os.path.join(targ_dir, "not_food")

for dir in [food_image_dir, not_food_image_dir]:
    os.makedirs(dir, exist_ok=True)

for dir, subdirs, files in os.walk(targ_dir):
    for subdir in subdirs:
        if subdir == "food_images":
            food_image_dir_path = os.path.join(dir, subdir)
            move_images(source=food_image_dir_path, destination=food_image_dir)
        elif subdir == "not_food_images":
            not_food_image_dir_path = os.path.join(dir, subdir)
            move_images(source=not_food_image_dir_path, destination=not_food_image_dir)

for dir in [food_image_dir, not_food_image_dir]:
    print(f"[INFO] Total images in {dir}: {len(os.listdir(dir))}")
