import os
import cv2
import csv
import glob
import random
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms

from PIL import Image


train_transform = transforms.Compose([
		transforms.RandomResizedCrop(128),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	])

test_transform = transforms.Compose([
		transforms.Resize(128),
		transforms.CenterCrop(128),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	])


class CiWork5(Dataset):
	def __init__(self, root_dir, pose_dir, size=128, mode='train'):
		super(CiWork5, self).__init__()

		self.data_dir = root_dir
		self.pose_dir = pose_dir
		self.size = size

		self.mode = mode
		self.cls_dim = 128
		self.max_object_num = 2

		self.default_obj = torch.tensor(np.zeros((3, self.size, self.size))).float()
		self.object_dir = './data/dataset_objects/' + self.mode + '/' 

		if mode == 'train':
			self.raw_images, self.pose_images, self.labels = self.load_csv('train/train.csv', mode)
		elif mode == 'eval':
			self.raw_images, self.pose_images, self.labels = self.load_csv('eval/eval.csv', mode)
		elif mode == 'test':
			self.raw_images, self.pose_images, self.labels = self.load_csv('test/test.csv', mode)


	def load_csv(self, file, mode):
		raw_images, pose_images, labels = [], [], []
		file_path = self.data_dir + mode
		pose_path = self.pose_dir + mode

		with open(os.path.join(self.data_dir, file)) as f:
			reader = csv.reader(f)
			for row in reader:
				if not row: 
					continue

				image, label = row

				if image == '' or label == '':
					continue

				raw_images.append(os.path.join(file_path, image))
				pose_images.append(os.path.join(pose_path, image))
				labels.append(int(label))

		assert len(raw_images) == len(labels)

		return raw_images, pose_images, labels


	def load_obj(self, path):
		target_labels = [0, 67, 62, 63] # Human, Phone, Computer, Book
		obj_images, obj_cls = [], []

		name = path.split('\\')[1]
		ddir = self.object_dir + name

		if os.path.exists(ddir):
			names = os.listdir(ddir)

			for name in names:
				img = Image.open(ddir + '/' + name).convert('RGB')
				label = int(name.split('.')[0])
				if label not in target_labels:
					continue
				obj_images.append(img)
				obj_cls.append(torch.tensor(label))

		return obj_images, obj_cls


	def __getitem__(self, idx):
		raw_image, pose_image, label = self.raw_images[idx], self.pose_images[idx], self.labels[idx]
		# Load object features and labels
		obj_images, obj_cls = self.load_obj(self.raw_images[idx])

		raw_image = Image.open(raw_image).convert('RGB')
		pose_image = Image.open(pose_image).convert('RGB')

		if self.mode == 'train' or self.mode == 'eval':
			raw_image, pose_image = train_transform(raw_image), train_transform(pose_image)
			for index in range(len(obj_images)):
				obj_images[index] = train_transform(obj_images[index]).float()
		else:
			raw_image, pose_image = test_transform(raw_image), test_transform(pose_image)
			for index in range(len(obj_images)):
				obj_images[index] = test_transform(obj_images[index]).float()

		label = torch.tensor(label)

		# Pad Or Trim
		while True:
			if len(obj_images) < self.max_object_num:
				obj_images.append(self.default_obj)
				obj_cls.append(torch.tensor(-1))
			else:
				break

		obj_images = obj_images[:self.max_object_num]
		obj_cls = obj_cls[:self.max_object_num]

		obj_cls_onehot = []
		# One-hot encoding
		for index in range(self.max_object_num):
			if obj_cls[index].item() >= 0:
				cls_feats = torch.eye(self.cls_dim).index_select(0, obj_cls[index]).squeeze()
			else:
				cls_feats = torch.zeros((self.cls_dim))
			obj_cls_onehot.append(cls_feats)

		return raw_image, pose_image, obj_images, obj_cls_onehot, label


	def __len__(self):
		return len(self.raw_images)


def init_dataset(opt):
	trn_dataset = CiWork5(root_dir='./data/dataset_shuffled/', pose_dir='./data/dataset_shuffle_pose/', mode='train')
	val_dataset = CiWork5(root_dir='./data/dataset_shuffled/', pose_dir='./data/dataset_shuffle_pose/', mode='eval')
	tst_dataset = CiWork5(root_dir='./data/dataset_shuffled/', pose_dir='./data/dataset_shuffle_pose/', mode='test')

	trn_dataloader = DataLoader(trn_dataset, batch_size=opt.batch_size, shuffle=True)
	val_dataloader = DataLoader(val_dataset, batch_size=opt.batch_size//2, shuffle=True)
	tst_dataloader = DataLoader(tst_dataset, batch_size=opt.batch_size//2, shuffle=True)

	return trn_dataloader, val_dataloader, tst_dataloader