"""
Script that converts images from PNG format to JPEG format.

JPEG images can preserve enough quality and are way smaller compared to PNG images.
"""

import glob
import pathlib

from PIL import Image

PATHFILE_TO_SAVE = "../image_data/labeled_images-jpeg"
PNG_PATHFILE = "../image_data/labeled_images"

pathlib.Path(PATHFILE_TO_SAVE).mkdir(parents=True, exist_ok=True)

for image in glob.iglob(f"{PNG_PATHFILE}/*.png"):
    im = Image.open(image)
    im = im.convert('RGB')
    name = image.removeprefix(f"{PNG_PATHFILE}/").removesuffix(".png")
    im.save(f"{PATHFILE_TO_SAVE}/{name}.jpeg", quality=90)
