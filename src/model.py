import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models


# Baseline: Resnet50
class WorkStateClsModel(nn.Module):
	def __init__(self):
		super().__init__()
		self.device = 'cuda:0'

		self.global_dim = 2048
		self.local_dim = 2048
		self.cls_dim = 128

		self.object_num = 5

		# Resnet50 Backbone -- Global Apperence Feature Extractor
		self.backbone_apperance = models.resnet50(pretrained=True)
		num_fts = self.backbone_apperance.fc.in_features

		self.backbone_apperance.fc = nn.Sequential()

		# Resnet50 Backbone -- Local Object Feature Extractor
		self.backbone_object = models.resnet50(pretrained=True)
		self.backbone_object.fc = nn.Sequential()

		# Resnet50 Backbone -- Pose Feature Classifier
		self.backbone_pose = models.resnet50(pretrained=True)
		num_fts = self.backbone_pose.fc.in_features
		self.backbone_pose.fc = nn.Linear(num_fts, 5)

		# Fusion factors
		self.apperance_f = 1.0
		self.sematic_f = 1.0
		self.pose_f = 0.0

		#self.feat_fc = nn.Linear(self.global_dim + self.local_dim  * self.object_num, 5)
		self.feat_fc = nn.Linear(self.global_dim, 5)
		self.sematic_fc = nn.Linear(self.cls_dim * self.object_num, 5)
		# Softmax
		self.softmax = nn.Softmax(dim=1)


	def forward(self, x_a, x_p, x_o, x_c):
		apperance_feats = self.backbone_apperance(x_a)
		'''
		object_feats, cls_feats = None, None
		for index in range(self.object_num):
			if index == 0:
				object_feats = self.backbone_object(x_o[index].to(self.device))
				cls_feats = x_c[index].to(self.device)
			else:
				object_feats = torch.cat([object_feats, self.backbone_object(x_o[index].to(self.device))], dim=-1)
				# One-hot encoding
				cls_feats = torch.cat([cls_feats, x_c[index].to(self.device)], dim=-1)
		'''
		
		# Object Features -- Pre Fusion
		#feats = torch.cat([apperance_feats], dim=-1)
		feats = apperance_feats
		apperance_pred = self.feat_fc(feats)

		# Object Label Features  -- Post Fusion
		#cls_pred = self.sematic_fc(cls_feats)

		# Pose Features -- Post Fusion
		#pose_pred = self.backbone_pose(x_p)

		#x = self.apperance_f * apperance_pred + self.pose_f * pose_pred + self.sematic_f * cls_pred
		x = self.apperance_f * apperance_pred

		x = self.softmax(x)

		return x