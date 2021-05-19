import os
import random
import logging
from collections import defaultdict
from shutil import copyfile

# logger configuration
LOG_FORMAT = "%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)

# dataset info
ROOT = "../dataset"
DATASET = "full"
PROCEDURE = ["train","eval","test"]

LABELS = ['Sleeping', 'Dazing', 'Playing Phone', 'Reading Book', 'Working On Computer']

TRAIN_RATE = 0.6
TEST_RATE = 0.2 # EVAL_RATE

if not os.path.exists(os.path.join(ROOT, DATASET)):
    logging.warning(f"Dataset does not exist!!!")
    exit(0)

# check split data folder
for proc in PROCEDURE:
    if not os.path.exists(os.path.join(ROOT, proc)):
        os.mkdir(os.path.join(ROOT, proc))

# retrieve all images
images = os.listdir(os.path.join(ROOT, DATASET))
logging.info(f"{len(images)} images are found in [{os.path.join(ROOT, DATASET)}].")

# aggregate images with label
dataset = defaultdict(list)
for image in images:
    temp = image.split("_")
    dataset[temp[0]].append(temp[1])

logging.info(f"random shuffle images.")
for label in dataset.keys():
    random.shuffle(dataset[label])

logging.info(f"Spliting images into train/eval/test : {TRAIN_RATE}/{TEST_RATE}/{TEST_RATE}")
for label in dataset.keys():
    n = len(dataset[label])
    train = int(n * TRAIN_RATE)
    test = int(n * TEST_RATE)

    # print(dataset[label][-test:])
    for data in dataset[label][:train]:
        src = os.path.join(ROOT, DATASET, label+"_"+data)
        dst = os.path.join(ROOT, PROCEDURE[0], label+"_"+data)
        copyfile(src, dst)

    # print(dataset[label][-test:])
    for data in dataset[label][train:-test]:
        src = os.path.join(ROOT, DATASET, label+"_"+data)
        dst = os.path.join(ROOT, PROCEDURE[1], label+"_"+data)
        copyfile(src, dst)

    # print(dataset[label][train:-test])
    for data in dataset[label][-test:]:
        src = os.path.join(ROOT, DATASET, label+"_"+data)
        dst = os.path.join(ROOT, PROCEDURE[2], label+"_"+data)
        copyfile(src, dst)

# Count static for dataset
n_train = len(os.listdir(os.path.join(ROOT, PROCEDURE[0])))
n_eval = len(os.listdir(os.path.join(ROOT, PROCEDURE[1])))
n_test = len(os.listdir(os.path.join(ROOT, PROCEDURE[2])))

logging.info(f"\n    DataSet Static\n\
    Train image: {n_train}\n\
    Eval image: {n_eval}\n\
    Test image: {n_test}\n ")