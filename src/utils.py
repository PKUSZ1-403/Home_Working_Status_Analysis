import os
import csv
import glob
import random

import torch
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms

from PIL import Image

train_transform = transforms.Compose([
		lambda x : Image.open(x).convert('RGB'),
		transforms.RandomResizedCrop(128),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	])

test_transform = transforms.Compose([
		lambda x : Image.open(x).convert('RGB'),
		transforms.Resize(128),
		transforms.CenterCrop(128),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	])

class CiWork5(Dataset):
	def __init__(self, root_dir, resize=256, mode='train'):
		super(CiWork5, self).__init__()

		self.data_dir = root_dir
		self.resize = resize

		self.mode = mode

		if mode == 'train':
			self.images, self.labels = self.load_csv('train/train.csv', mode)
		elif mode == 'eval':
			self.images, self.labels = self.load_csv('eval/eval.csv', mode)
		elif mode == 'test':
			self.images, self.labels = self.load_csv('test/test.csv', mode)


	def load_csv(self, file, mode):
		images, labels = [], []
		file_path = self.data_dir + mode

		with open(os.path.join(self.data_dir, file)) as f:
			reader = csv.reader(f)
			for row in reader:
				if not row: 
					continue

				image, label = row

				if image == '' or label == '':
					continue

				images.append(os.path.join(file_path, image))
				labels.append(int(label))

		assert len(images) == len(labels)

		return images, labels


	def __getitem__(self, idx):
		image, label = self.images[idx], self.labels[idx]

		if self.mode == 'train' or self.mode == 'eval':
			image = train_transform(image)
		else:
			image = test_transform(image)
		label = torch.tensor(label)

		return image, label


	def __len__(self):
		return len(self.images)


def init_dataset(opt):
	trn_dataset = CiWork5(root_dir='./data/dataset/', mode='train')
	val_dataset = CiWork5(root_dir='./data/dataset/', mode='eval')
	tst_dataset = CiWork5(root_dir='./data/dataset/', mode='test')

	trn_dataloader = DataLoader(trn_dataset, batch_size=opt.batch_size, shuffle=True)
	val_dataloader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=True)
	tst_dataloader = DataLoader(tst_dataset, batch_size=opt.batch_size, shuffle=True)

	return trn_dataloader, val_dataloader, tst_dataloader