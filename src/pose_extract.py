import os
import cv2
import matplotlib.pyplot as plt
import copy
import numpy as np
from PIL import Image

from libs import model
from libs import util
from libs.body import Body
from libs.hand import Hand

class PoseExtract:
    def __init__(self, body_model='./models/body_pose_model.pth', hand_model='./models/hand_pose_model.pth'):
        self.body_estimation = Body(body_model)
        self.hand_estimation = Hand(hand_model)

        self.background_image = './imgs/black.jpg'
        self.background = cv2.imread(self.background_image)

    def pose_extract(self, input):
        background = cv2.resize(self.background, (input.shape[1], input.shape[0]))

        candidate, subset = self.body_estimation(input)
        canvas = copy.deepcopy(background)
        canvas = util.draw_bodypose(canvas, candidate, subset)
        # detect hand
        hands_list = util.handDetect(candidate, subset, input)

        all_hand_peaks = []
        for x, y, w, is_left in hands_list:
            
            peaks = self.hand_estimation(input[y:y+w, x:x+w, :])
            peaks[:, 0] = np.where(peaks[:, 0]==0, peaks[:, 0], peaks[:, 0]+x)
            peaks[:, 1] = np.where(peaks[:, 1]==0, peaks[:, 1], peaks[:, 1]+y)
            
            all_hand_peaks.append(peaks)

        canvas = util.draw_handpose(canvas, all_hand_peaks)
        pose_image = canvas[:, :, [2, 1, 0]]

        # cv2 to PIL
        pose_image = Image.fromarray(cv2.cvtColor(pose_image, cv2.COLOR_BGR2RGB))

        return pose_image


if __name__ == '__main__':
    # Perform dataset pose pre-extract
    pose_extracter = PoseExtract()
    ddir = './data/dataset_shuffled/'
    new_ddir = './data/dataset_shuffle_pose'

    for mode in ['train', 'test', 'eval']:
        ppath = ddir + mode
        names = os.listdir(ppath)

        for name in names:
            if name.split('.')[0] == mode: continue
            
            pic_path = ppath + '/' + name
            image = cv2.imread(pic_path)
            pose_image = pose_extracter.pose_extract(image)
            pose_image.save(new_ddir + '/' + mode + '/' + name)

        print('==' + mode)