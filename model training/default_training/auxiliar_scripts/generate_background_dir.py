"""
Background images proved to be a challenge during the data splitting process
if they are present in the same directory of labeled images.

Because these images don't have labels (.txt files) related to them,
the set of labeled images and related labels could be incorrectly affected
by background images.

This script was created to ensure that background images could have their own directory and be part of a specific
splitting process without affecting the remaining dataset.
"""

import glob
import os
import shutil
from pathlib import Path

IMG_PATH = "../image_data/labeled_images"
LABEL_PATH = "../image_data/labels"
BACKGROUND_PATH = "../image_data/background_images"

# Creates a directory for background images if it doesn't exist
Path(BACKGROUND_PATH).mkdir(parents=True, exist_ok=True)

bg_count = 0
for image in glob.iglob(f"{IMG_PATH}/*.jpeg"):

    # Gets image name
    name = image.split(IMG_PATH)[1].split(".jpeg")[0]
    
    # Checks if there is a correspondent txt label file related to the image
    label_exists = os.path.isfile(f"{LABEL_PATH}/{name}.txt")
    
    # If the label doesn't exist, the image is a background image (no labels related to it)
    if not label_exists:
        try: # Move background image to an appropriate directory
            bg_count += 1
            shutil.move(image, BACKGROUND_PATH)
        except:
            print(f)
            assert False

print(f"\n{bg_count} background images moved from \"{IMG_PATH}\" to \"{BACKGROUND_PATH}\".\n")
