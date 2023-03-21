import argparse
import fnmatch
import glob
import numpy as np
import os
import pathlib
import torch

from random import shuffle
from shutil import move
from scipy.stats import entropy

SAVE_PATH = "../datasets/dusk_enemies"
MODEL_PATH = "../yolov5/dusk_incremental/train_256_/weights/best.pt"
DATA_PATH = ".."

def parse_arguments():
    """ Parses arguments from command line. """
    parser = argparse.ArgumentParser(description="Extract frames from videos")

    parser.add_argument("-i", "--images", type=int, help="Total number of images of the current model", required=True)
    parser.add_argument("-t", "--training", type=bool, help="Type of training (True=incremental, False=from zero)", required=True)
    parser.add_argument("-m", "--model", type=str, default=MODEL_PATH, help="Path to the model")
    parser.add_argument("-s", "--save", type=str, default=SAVE_PATH, help="Path to save analyzed images")
    
    return parser.parse_args()


def get_number_of_existent_imgs(path):
    """ Returns the number of images already in the dataset. """
    return len(fnmatch.filter(os.listdir(path), '*.jpeg'))


def check_if_directory_exists(path):
    """ Checks if directory exists. If not, creates it."""
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


def calculate_norm_entropy(probabilities):
    """ Calculates the normalized entropy of a list of probabilities. """
    
    probabilities = np.array(probabilities)

    # By default, YOLO does not consider the confidence of background images.
    #
    # Because of that, we will append the background confidence to the array so that
    # the total confidence equals 1 and the entropy is calculated correctly.
    np.append(probabilities, 1 - probabilities.sum())
    
    return entropy(probabilities, base=2) / np.log2(len(probabilities))


def main():
    global SAVE_PATH
    global DATA_PATH
    args = parse_arguments()

    # Checks if the data will be added to the incremental or from_zero directory
    if args.training is True:
        SAVE_PATH = os.path.join(SAVE_PATH, "incremental")
        DATA_PATH = os.path.join(DATA_PATH, "images_incremental")
    else:
        SAVE_PATH = os.path.join(SAVE_PATH, "from_zero")
        DATA_PATH = os.path.join(DATA_PATH, "images_from_zero")

    # Calculating images to add to the training set
    added_imgs = 0
    imgs_folder = glob.glob(os.path.join(DATA_PATH, "*.jpeg"))
    imgs_to_add = int((args.images * 0.8)) - get_number_of_existent_imgs('../datasets/dusk_enemies/incremental/images/train')
    max_train_bg_imgs = imgs_to_add * 0.1

    shuffle(imgs_folder)

    model = torch.hub.load('../yolov5', 'custom', source='local', path = MODEL_PATH, force_reload = True)
    model.conf = 0.5

    for img in imgs_folder:
        if added_imgs >= imgs_to_add:
            break
            
        img_name = img.split("/")[-1].split(".")[0]

        results = model(img)

        # For each image, if no predictions were made according to the model threshold,
        # the image is considered a background image and moved to the background_images folder.
        if len(results.xywhn[0]) < 1:

            # According to the YOLOv5 guidelines, the background images should not exceed 10% of the dataset.
            # If more than 10% of the images are background images (checked by max_train_bg_imgs),
            # the image is ignored.
            if added_imgs > max_train_bg_imgs:
                print("Background image ignored: ", img.split("/")[-1])
            else:
                print("Background image: ", img.split("/")[-1])
                move(img, os.path.join(SAVE_PATH, "images", "train"))
                added_imgs += 1
        else:
            labels = results.xywhn[0][:, -1].numpy()
            all_cls_conf = results.all_cls_conf[0][:].numpy()
            bb_coords = results.xywhn[0][:, :4].numpy()

            ignored_img = False

            # Ignores the image if the normalized entropy of the predictions is greater than 0.4.
            for conf in all_cls_conf:
                entropy = calculate_norm_entropy(conf)
                if entropy >= 0.4:
                    print("Ignored image: ", img.split("/")[-1])
                    ignored_img = True
                    break
            
            if not ignored_img:
                print("Labeled image: ", img.split("/")[-1])
                move(img, os.path.join(SAVE_PATH, "images", "train"))

                # Runs the loop for each prediction
                for class_label, bounding_box in zip(labels, bb_coords):
                    with open(os.path.join(SAVE_PATH, "labels", "train", f"{img_name}.txt"), "a+") as yolo_file:
                        yolo_file.write(f"{int(class_label)} {bounding_box[0]:.6f} {bounding_box[1]:.6f} {bounding_box[2]:.6f} {bounding_box[3]:.6f}\n")
                added_imgs += 1
    
    print(f"A total of {added_imgs} images were added to the training set.")

if __name__ == "__main__":
    main()