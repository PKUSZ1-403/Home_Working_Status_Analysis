import os
import cv2
import argparse

import torch
from torchvision import transforms

from tkinter import * 
from tkinter import filedialog
from PIL import Image, ImageTk

from model import WorkStateClsModel as Model


transform = transforms.Compose([
	transforms.Resize(128),
	transforms.CenterCrop(128),
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

CiWork5 = ['Sleeping', 'Dazing', 'Playing Phone', 'Reading Book', 'Working On Computer']


class MainWindow(Frame):
	def __init__(self, master, model):
		Frame.__init__(self, master)
		self.master = master
		self.model = model.cuda()

		self.state_label = ''
		self.score = 0.0

		self.btn = Button(self, text='Upload Image', fg='black', bg='gray', command=self.pressUpload)
		self.btn.place(x=380, y=650, height=40, width=150)

		self.label = Label(self, text="Current Image's Working State :", font=('Times New Roman', 18, 'bold'))
		self.label.place(x=100, y=500, height=100, width=350)

		self.state = Label(self, text="__________________", font=('Times New Roman', 18, 'bold'))
		self.state.place(x=460, y=500, height=100, width=400)

		self.hint = Label(self, text="", font=('Times New Roman', 15, 'bold'))
		self.hint.place(x=210, y=580, height=50, width=500)

		default_img = Image.open('imgs/default.jpg').resize((450, 450))
		render = ImageTk.PhotoImage(default_img)
		self.image = Label(self, image=render)
		self.image.image = render
		self.image.place(x=230, y=30, height=450, width=450)

		self.pack(fill=BOTH, expand=1)

	def pressUpload(self):
		# Upload Image
		selectImg = filedialog.askopenfilename()

		if selectImg == '': 
			return

		# Show Image
		image = Image.open(selectImg).convert('RGB')
		rawImg = image # store raw image

		img = image.resize((450, 450))
		render = ImageTk.PhotoImage(img)

		self.image = Label(self, image=render)
		self.image.image = render
		self.image.place(x=230, y=30, height=450, width=450)

		# Change State
		self.state_label, self.score = self.predict(rawImg)
		self.state['text'] = CiWork5[self.state_label] + ' (' + str(self.score)[:5] + ')'
		if self.state_label <= 2:
			self.state['fg'] = 'red'
			self.hint['fg'] = 'red'
			self.hint['text'] = "Please don't be distracted while working!"
		else:
			self.state['fg'] = 'green'
			self.hint['fg'] = 'green'
			self.hint['text'] = "Focus on working...."

	def predict(self, image):
		image = transform(image)

		self.model.eval()
		image = image.cuda()

		image = image.unsqueeze(0)

		pred = self.model(image)
		label = pred.argmax().item()

		prob = pred.max().item() / pred.sum().item()

		return label, prob


def init_model(opt):
	model = Model()
	model = model.cuda() if opt.cuda else model
	return model


def load_model(opt):
	best_model_path = os.path.join(opt.ckpt_dir, 'best_model_path')
	best_model_state = torch.load(best_model_path)

	model = init_model(opt)
	model.load_state_dict(best_model_state)

	return model


def demo(window):
	# Run all test images demo
	ddir = './data/dataset/train'
	names = os.listdir(ddir)

	correct = [0, 0, 0, 0, 0]
	total = [0, 0, 0, 0, 0]

	ddict = {'Sleeping':0, 'Dazing':1, 'Playing Phone':2, 'Reading Book':3, 'Working On Computer':4}

	for name in names:
		if name.split('.')[0] == 'train': continue
		image = Image.open(ddir + '/' + name).convert('RGB')
		label, _ = window.predict(image)
		gt_label = ddict[name.split('_')[0]]
		if name.split('_')[0] == CiWork5[label]:
			correct[label] += 1
		else:
			print(name + '    ' + CiWork5[label])
		total[gt_label] += 1
	
	print(correct)
	print(total)
	print(CiWork5)

	for i in range(5):
		print(correct[i] / total[i])


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--ckpt_dir', type=str, default='checkpoints/')
	parser.add_argument('--cuda', action='store_true')
	parser.add_argument('--device', type=int, default=0)

	options = parser.parse_args()
	model = load_model(options)

	# Python tkinter GUI
	root = Tk()
	root.geometry("900x750") # 600x500
	root.title('Home Working State Detection')
	window = MainWindow(root, model=model)

	#demo(window)

	root.mainloop()


if __name__ == '__main__':
	main()