import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models


# Baseline: LeNet-5
class WorkStateClsModel(nn.Module):
	def __init__(self):
		super(self, WorkStateClsModel).__init__()

		self.backbone = models.resnet101(pretrained=True)
		num_fts = self.backbone.fc.in_features

		self.backbone.fc = nn.Linear(num_fts, 5)


	def forward(self, x):
		x = self.backbone(x)

		return x