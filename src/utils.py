import os
import csv
import glob
import random

import torch
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms

from PIL import Image


class CiWork5(Dataset):
	def __init__(self, root_dir, resize=256, mode='train'):
		super(CiWork5, self).__init__()

		self.root_dir = root_dir
		self.resize = resize

		self.images, self.labels = self.load_csv('labels.csv')

		if mode == 'train': # 70% = 0% -> 70%
			self.images = self.images[:int(0.7 * len(self.images))]
			self.labels = self.labels[:int(0.7 * len(self.labels))]
		elif mode == 'val': # 10% = 70% -> 80%
			self.images = self.images[int(0.7 * len(self.images)):int(0.8 * len(self.images))]
			self.labels = self.labels[int(0.7 * len(self.labels)):int(0.8 * len(self.images))]
		elif mode == 'test': # 20% = 80% -> 100%
			self.images = self.images[int(0.8 * len(self.images)):]
			self.labels = self.labels[int(0.8 * len(self.labels)):]


	def load_csv(self, file):
		images, labels = [], []
		with open(os.path.join(self.root_dir, file)) as f:
			reader = csv.reader(f)
			for row in reader:
				image, label = row

				images.append(os.path.join(self.root_dir, image))
				labels.append(int(label))

		assert len(images) == len(labels)

		return images, labels


	def __getitem__(self, idx):
		image, label = self.images[idx], self.labels[idx]

		transform = transforms.Compose([
			lambda x : Image.open(x).convert('RGB'),
			transforms.Resize((int(self.resize * 1.0), int(self.resize * 1.0))),
			transforms.RandomRotation(15),
			transforms.CenterCrop(self.resize),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		])

		image = transform(image)
		label = torch.tensor(label)

		return image, label


	def __len__(self):
		return len(self.images)


def init_dataset(opt):
	trn_dataset = CiWork5(root_dir='dataset/', mode='train')
	val_dataset = CiWork5(root_dir='dataset/', mode='val')
	tst_dataset = CiWork5(root_dir='dataset/', mode='test')

	trn_dataloader = DataLoader(trn_dataset)
	val_dataloader = DataLoader(val_dataset)
	tst_dataloader = DataLoader(tst_dataset)

	return trn_dataloader, val_dataloader, tst_dataloader