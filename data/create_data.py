import cv2 as cv
import os
import logging
import argparse

# command argument parser
parser = argparse.ArgumentParser()
parser.add_argument("-r", "--rate", type=int, default=20, help="Video frame sample rate")
args = parser.parse_args()

# logger configuration
LOG_FORMAT = "%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)

# sample rate
FRAME_RATE = args.rate

# Basic data info
LABELS = ['Sleeping', 'Dazing', 'Playing Phone', 'Reading Book', 'Working On Computer']

VIDEO_PATH = "video"
DATA_PATH = "../dataset/full"

# check dataset folder
if not os.path.exists(DATA_PATH):
    os.mkdir(DATA_PATH)

for label in LABELS:
    logging.info(f"Extracting {label} image from video...")

    label_cnt = 0
    path = os.path.join(VIDEO_PATH, label)

    for file in os.listdir(path):
        logging.info(f"Extracting video: {file}...")
        video = cv.VideoCapture(os.path.join(VIDEO_PATH, label, file))
        ret, frame = video.read()
        cnt = 1
        while ret:
            if cnt % FRAME_RATE == 0:
                cv.imwrite(os.path.join(DATA_PATH, label+"_"+str(label_cnt)+".jpg"), frame)
                label_cnt += 1
            cnt += 1
            ret, frame = video.read()    

        video.release()
    logging.info(f"[{str(label_cnt)}] {label} images are extracted")
