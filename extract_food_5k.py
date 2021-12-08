"""
Extracts images from Food-5k into "food" and "not_food" categories,
training, evaluation and test sets are ignored.

Original data downloaded here: https://www.kaggle.com/binhminhs10/food5k
"""
import os
import pathlib
import shutil
import uuid

# Check to see if path exists, if not exit
assert os.path.exists(
    "data/Food-5k"
), "The Food-5k dataset is not downloaded, please get it from: \
     https://www.kaggle.com/binhminhs10/food5k"

# Create dir to extract to (images go here)
targ_path = "data/Food5k_extracted"
not_food_image_path = os.path.join(targ_path, "not_food_images")
food_image_path = os.path.join(targ_path, "food_images")

# If targ path already exists, remove it
try:
    os.makedirs(targ_path)
except:
    shutil.rmtree(targ_path)

# Recreate new directories for targets
os.makedirs(targ_path)
os.makedirs(not_food_image_path)
os.makedirs(food_image_path)

# Get path names
paths = list(pathlib.Path("data/Food-5k").glob("*/*.jpg"))
print(f"[INFO] Total images found: {len(paths)}")

# Loop through paths
not_food_image_count = 0
food_image_count = 0
for i, path in enumerate(paths):
    filename = path.name
    if str(filename[0]) == "0":
        not_food_image_count += 1
        # Need to rename file due to images in train/val/test having same name
        new_file_name = pathlib.Path(str(uuid.uuid4()) + "_" + str(filename))
        dest_path = pathlib.Path(not_food_image_path, new_file_name)
        shutil.copy2(path, dest_path)

        if i % 250 == 0:
            print(f"[INFO] Copying {path} to {dest_path}...")
            print(f"[INFO] Total not food images moved: {not_food_image_count}")

    else:
        food_image_count += 1
        new_file_name = pathlib.Path(str(uuid.uuid4()) + "_" + str(filename))
        dest_path = pathlib.Path(food_image_path, new_file_name)
        shutil.copy2(path, dest_path)

        if i % 250 == 0:
            print(f"[INFO] Copying {path} to {dest_path}...")
            print(f"[INFO] Total food images moved: {food_image_count}")

print(f"\n[INFO] ---- Finished Copying ----\n")
print(f"[INFO] Total not food images moved: {len(os.listdir(not_food_image_path))}")
print(f"[INFO] Total food images moved: {len(os.listdir(food_image_path))}")
