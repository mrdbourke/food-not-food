"""
Script to download data from Food101 dataset.

Data has been preprocessed into:

101_food_classes_all_data/
    train/
        apple_pie/
            134.jpg
            ...
        ...
    test/
        apple_pie/
            38795.jpg
            ...
        ...
"""
import os
import shutil
import requests
import zipfile

from shutil import move, copy2
from tqdm import tqdm


def unzip_data(filename):
    """
    Unzips filename into the current working directory.

    Args:
      filename (str): a filepath to a target zip folder to be unzipped.
    """
    zip_ref = zipfile.ZipFile(filename, "r")
    zip_ref.extractall()
    zip_ref.close()


def download(url: str, filename: str):
    """Prints progress bar for downloading a file.

    Args:
        url (str): URL of target file to download.
        filename (str): description of file being downloaded (this will print out in the progress bar).
    """
    response = requests.get(url, stream=True)
    total_downloaded = int(response.headers.get("content-length", 0))
    with open(filename, "wb") as file, tqdm(
        desc=filename,
        total=total_downloaded,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)


# Download data
if not os.path.exists("*/101_food_classes_all_data.zip"):
    if not os.path.exists("*/101_food_classes_all_data"):
        zip_path = "101_food_classes_all_data.zip"
        print(f"[INFO] Downloading Food101 data...")
        download(
            url="https://storage.googleapis.com/ztm_tf_course/food_vision/101_food_classes_all_data.zip",
            filename=zip_path,
        )
        # requests.get(
        #     "https://storage.googleapis.com/ztm_tf_course/food_vision/101_food_classes_all_data.zip"
        # )
        print(f"[INFO] Food101 downloaded, unzipping...")
        unzip_data(zip_path)
        print(f"[INFO] Data unzipped, moving to data directory...")
        move("101_food_classes_all_data", "../data")
        print(f"[INFO] Data moved to data directory, deleting downloaded zip file...")
        os.remove(zip_path)
