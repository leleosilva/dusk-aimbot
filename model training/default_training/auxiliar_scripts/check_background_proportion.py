"""
Script that counts the number of background images by checking images that don't have labels (txt files)
related to it, without moving them to another directory.
It also shows the proportion of background images present in the whole dataset.

Useful for knowing the proportion of background images (YOLOv5 documentation recommends about 0-10% background images
to help reduce false positives (https://github.com/ultralytics/yolov5/wiki/Tips-for-Best-Training-Results#dataset).
"""

import glob
import os

IMG_PATH = "../image_data/labeled_images"
LABEL_PATH = "../image_data/labels"

bg_count = 0
img_count = 0
for image in glob.iglob(f"{IMG_PATH}/*.jpeg"):
    img_count += 1

    # Gets image name
    name = image.split(f"IMG_PATH/")[1].split(".jpeg")[0]
    
    # Checks if there is a correspondent txt label file related to the image
    label_exists = os.path.isfile(f"{LABEL_PATH}/{name}.txt")
    
    # If the label doesn't exist, the image is a background image (no labels related to it)
    if not label_exists:
        bg_count += 1

try:
    bg_proportion = round((bg_count / img_count) * 100, 1)
except:
    bg_proportion = 0.0
print(f"\nBackground images: {bg_count} of a total of {img_count} images ({bg_proportion}%).\n")
