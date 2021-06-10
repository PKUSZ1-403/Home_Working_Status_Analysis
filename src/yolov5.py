import cv2
from PIL import Image

import os

import torch
import torch.nn as nn
import torch.nn.functional as F


# Object extractor -- Yolov5s
class Yolov5:
	def __init__(self):
		super().__init__()
		self.object_extractor = torch.hub.load('ultralytics/yolov5', 'yolov5s')
		self.threshold = 0.5
		self.max_object_num = 5


	def object_extract(self, rawImg, mode, name):
		results = self.object_extractor(rawImg)
		result = results.xyxy[0]
		object_images, object_cls_vectors = [], []

		for bbox in result:
			bbox = bbox.cpu().numpy()
			conf = bbox[4].item()
			# Confidence score threshold
			if conf < self.threshold: 
				continue
			
			# Obtain object crop-images
			position = [int(bbox[i]) for i in range(4)]
			cropped = rawImg[position[1]:position[3], position[0]:position[2]]
			# Obtain object prediction
			label = int(bbox[5].item())
			# Store cropped image
			ddir = './data/dataset_objects/' + mode + '/' + name + '/'
			if not os.path.exists(ddir):
				os.makedirs(ddir)

			cv2.imwrite(ddir + str(label) + '.jpg', cropped)


if __name__ == '__main__':
    # Perform dataset objects pre-extract
    object_extracter = Yolov5()
    ddir = './data/dataset_shuffled/'
    
    for mode in ['train', 'test', 'eval']:
        ppath = ddir + mode
        names = os.listdir(ppath)

        for name in names:
            if name.split('.')[0] == mode: continue
            
            pic_path = ppath + '/' + name
            image = cv2.imread(pic_path)
            object_extracter.object_extract(image, mode, name)
            print('==' + pic_path)

        print('==' + mode)