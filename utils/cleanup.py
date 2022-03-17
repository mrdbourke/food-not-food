"""
Removes all image folders in data folder that aren't needed.
"""
import os
import shutil
from pathlib import Path

targ_path = Path("../data")

print(targ_path / "101_food_classes_all_data_extracted")

try:
    shutil.rmtree(targ_path / "101_food_classes_all_data_extracted")
except OSError as e:
    print(e.strerror)
