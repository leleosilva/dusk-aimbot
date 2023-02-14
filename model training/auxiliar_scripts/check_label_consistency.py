"""
Script that checks if all existent labels in LABEL_PATH are related
to an image in IMAGE_PATH.

Annotation softwares such as LabelImg may not delete label files (.txt)
as expected when rect boxes or even images are deleted.
This script may be useful on preventing this issue.
"""

import glob
import os

LABEL_PATH = "../image_data/labels"
IMAGE_PATH = "../image_data/labeled_images"

img_error = False
label_count = 0
for label in glob.iglob(f"{LABEL_PATH}/*.txt"):
    label_count += 1
    name = label.split(f"{LABEL_PATH}/")[1].split(".txt")[0]
    img_exists = os.path.isfile(f"{IMAGE_PATH}/{name}.jpeg")
    if not img_exists:
        img_error = True
        print(f"The image {name}.jpeg was not found.")

if label_count == 0 and not img_error: # No inconsistency nor labels
    print("\nThere are no labels present in the chosen path.\n")
elif not img_error: # Labels present and consistent
    print(f"\nAll {label_count} present labels are correctly related to an image.\n")
