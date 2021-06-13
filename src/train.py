import torch
import torch.nn as nn
from torch.optim import Adam

import os
import argparse
import numpy as np
from tqdm import tqdm

from utils import init_dataset
from model import WorkStateClsModel as Model


def init_model(opt):
	model = Model()
	model = model.cuda() if opt.cuda else model
	return model


def get_acc(y_pred, y):
	_, preds = y_pred.max(1)
	acc = torch.eq(preds, y.view_as(preds)).float().mean()

	return acc.item()


def train(opt, trn_dataloader, val_dataloader, model, optim):
	print('========== Start Training ==========')
	trn_loss, trn_acc, val_loss, val_acc = [], [], [], []
	best_acc = 0

	best_model_path = os.path.join(opt.ckpt_dir, 'best_model_path')
	last_model_path = os.path.join(opt.ckpt_dir, 'last_model_path')

	loss_fn = nn.CrossEntropyLoss()

	for epoch in range(opt.epochs):
		print('===== Epoch: {} ====='.format(epoch))
		trn_iter = iter(trn_dataloader)
		# Model set train
		model.train()
		model = model.cuda()

		for batch in tqdm(trn_iter):
			optim.zero_grad()
			x_a, x_p, x_o, x_c, y = batch
			x_a, x_p, y = x_a.to(opt.device), x_p.to(opt.device), y.to(opt.device)

			y_pred = model(x_a, x_p, x_o, x_c)

			loss = loss_fn(y_pred, y)

			loss.backward()
			optim.step()

			trn_loss.append(loss.item())
			trn_acc.append(get_acc(y_pred, y))
		avg_loss = np.mean(trn_loss)
		avg_acc = np.mean(trn_acc)
		print('Avg Train Loss: {}, Avg Train Acc: {}'.format(avg_loss, avg_acc))

		# Model validation
		val_iter = iter(val_dataloader)
		model.eval()

		for batch in tqdm(val_iter):
			x_a, x_p, x_o, x_c, y = batch
			x_a, x_p, y = x_a.to(opt.device), x_p.to(opt.device), y.to(opt.device)

			y_pred = model(x_a, x_p, x_o, x_c)

			loss = loss_fn(y_pred, y)

			val_loss.append(loss.item())
			val_acc.append(get_acc(y_pred, y))
		avg_loss = np.mean(val_loss)
		avg_acc = np.mean(val_acc)
		postfix = ' (Best)' if avg_acc >= best_acc else ' (Best: {})'.format(best_acc)

		print('Avg Val Acc: {}{}'.format(avg_acc, postfix))

		# Save model with best validation accuracy
		if avg_acc >= best_acc:
			torch.save(model.state_dict(), best_model_path)
			best_state = model.state_dict()
			best_acc = avg_acc

	torch.save(model.state_dict(), last_model_path)

	print('========== End Training ==========')

	return best_state


def test(opt, tst_dataloader, model):
	print('========== Start Testing ==========')
	avg_acc = []

	for epoch in range(10):
		tst_iter = iter(tst_dataloader)
		model.eval()

		for batch in tqdm(tst_iter):
			x_a, x_p, x_o, x_c, y = batch
			x_a, x_p, y = x_a.to(opt.device), x_p.to(opt.device), y.to(opt.device)

			y_pred = model(x_a, x_p, x_o, x_c)

			avg_acc.append(get_acc(y_pred, y))
	avg_acc = np.mean(avg_acc)
	print('Test Acc (Epochs = 10): {}'.format(avg_acc))

	print('========== End Testing ==========')

	return avg_acc


def main():
	'''
	Parse input parameters and Init settings
	'''
	parser = argparse.ArgumentParser()
	parser.add_argument('--ckpt_dir', type=str, default='checkpoints/')
	parser.add_argument('--epochs', type=int, default=50)
	parser.add_argument('--batch_size', type=int, default=36)
	parser.add_argument('--lr', type=float, default=1e-4)
	parser.add_argument('--cuda', action='store_true')
	parser.add_argument('--device', type=int, default=0)

	options = parser.parse_args()

	if torch.cuda.is_available() and not options.cuda:
		print("WARNING: You have a CUDA device, so you should probably run with --cuda")

	# Load dataset
	print('========== Start Loading Dataset ==========')
	trn_dataloader, val_dataloader, tst_dataloader = init_dataset(options)
	print('========== End Loading Dataset ==========')
	# Init model and optim
	model = init_model(options)
	optim = Adam(params=model.parameters(), lr=options.lr)
	# Train model
	
	train(opt=options,
		  trn_dataloader=trn_dataloader,
		  val_dataloader=val_dataloader, 
		  model=model,
		  optim=optim)
	# Test model with checkpoints
	last_model_path = os.path.join(options.ckpt_dir, 'last_model_path')
	last_state = torch.load(last_model_path)
	model.load_state_dict(last_state)
	print('Testing with last model...')
	test(opt=options,
		 tst_dataloader=tst_dataloader,
		 model=model)

	print('Testing with best model...')
	best_model_path = os.path.join(options.ckpt_dir, 'best_model_path')
	best_state = torch.load(best_model_path)
	model.load_state_dict(best_state)
	test(opt=options,
		 tst_dataloader=tst_dataloader,
		 model=model)


if __name__ == '__main__':
	main()