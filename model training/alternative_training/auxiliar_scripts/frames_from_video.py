import argparse
import cv2
import glob
import os
import pathlib
import sys

from datetime import datetime

SECONDS_PER_FRAME = 3
FRAME_COUNT = 0

SOURCE_PATH = "../videos"
SAVE_PATH = "../images"

def parse_arguments():
    """ Parses arguments from command line. """
    parser = argparse.ArgumentParser(description="Extract frames from videos")

    parser.add_argument("-f", "--format", type=str, default="mp4", help="Video format")
    parser.add_argument("--source", type=str, default=SOURCE_PATH, help="Path to video files")
    parser.add_argument("--save", type=str, default=SAVE_PATH, help="Path to save frames")
    parser.add_argument("--seconds", type=int, default=SECONDS_PER_FRAME, help="Number of seconds between frames")
    
    return parser.parse_args()

def check_if_directory_exists(path):
    """ Checks if directory exists. If not, creates it."""
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

def generate_image_name():
        """ Generates image name in format YYYY-MM-DD_HH:MM:SS-sss (year-month-day_hour-minute-second-millisecond) """
        now_str = datetime.now().isoformat(sep='_').replace(':', '-').replace('.', '-')
        screenshot_name = f"{now_str}"
        return screenshot_name 

def get_video_frames(video, video_name):
    global FRAME_COUNT
    
    video_capture = cv2.VideoCapture(video)
    
    if not video_capture.isOpened():
        print("Could not load video")
        sys.exit(1)

    current_frame = 0
    print(f"Started extracting frames from '{video_name}'...")
    while True:

        # CAP_PROP_POS_MSEC gives the current position of the video in miliseconds
        # By multiplying it by 1000, it is possible to control the timestamp, in seconds
        video_capture.set(cv2.CAP_PROP_POS_MSEC, (current_frame * 1000))
        
        read_result, frame = video_capture.read()
        print ('New frame: ', read_result)

        if not read_result: # Video does not have any more frames to be read
            print(f"Done extracting frames. {FRAME_COUNT} frames were extracted in total.")
            break
        
        cv2.imwrite(f"{os.path.join(SAVE_PATH, generate_image_name())}.jpeg", frame)
        current_frame += SECONDS_PER_FRAME
        FRAME_COUNT += 1

def main():
    args = parse_arguments()

    for video in glob.iglob(os.path.join(SOURCE_PATH, f"*.{args.format}")):
        video = "/home/leleo/dusk-aimbot/model training/incremental_training/videos/dusk_farm2.mp4"
        video_name = video.split("/")[-1].split(".")[0]
        check_if_directory_exists(SAVE_PATH)
        get_video_frames(video, video_name)

if __name__ == "__main__":
    main()