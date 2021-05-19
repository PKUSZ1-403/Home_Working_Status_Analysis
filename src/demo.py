import os
import argparse

import torch
from torchvision import transforms

from tkinter import * 
from tkinter import filedialog
from PIL import Image, ImageTk

from model import WorkStateClsModel as Model


img_size = 256

transform = transforms.Compose([
	lambda x : Image.open(x).convert('RGB'),
	transforms.Resize((int(img_size * 1.0), int(img_size * 1.0))),
	transforms.RandomRotation(15),
	transforms.CenterCrop(img_size),
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

CiWork5 = ['Sleeping', 'Dazing', 'Playing Phone', 'Reading Book', 'Working On Computer']

def predict(model, image):
	return 0, 1.0
	image = transform(image)

	pred = model(image)
	label = pred.argmax().item()
	score = pred.max().item()

	return label, score


class MainWindow(Frame):
	def __init__(self, master):
		Frame.__init__(self, master)
		self.master = master

		# self.model = load_model(options)
		self.model = None
		self.state_label = ''
		self.score = 0.0

		self.btn = Button(self, text='Upload Image', fg='black', bg='gray', command=self.pressUpload)
		self.btn.place(x=240, y=420, height=40, width=100)

		self.label = Label(self, text="Current Image's Working State is:", font=('Times New Roman', 14, 'bold'))
		self.label.place(x=30, y=340, height=50, width=280)

		self.state = Label(self, text="__________________", font=('Times New Roman', 14, 'bold'))
		self.state.place(x=330, y=340, height=50, width=230)

		self.hint = Label(self, text="", font=('Times New Roman', 12, 'bold'))
		self.hint.place(x=150, y=380, height=30, width=300)

		default_img = Image.open('imgs/default.jpg').resize((300, 300))
		render = ImageTk.PhotoImage(default_img)
		self.image = Label(self, image=render)
		self.image.image = render
		self.image.place(x=150, y=30, height=300, width=300)

		self.pack(fill=BOTH, expand=1)

	def pressUpload(self):
		# Upload Image
		selectImg = filedialog.askopenfilename()

		if selectImg == '': 
			return

		# Show Image
		image = Image.open(selectImg)
		img = image.resize((300, 300))
		render = ImageTk.PhotoImage(img)

		self.image = Label(self, image=render)
		self.image.image = render
		self.image.place(x=150, y=30, height=300, width=300)

		# Change State
		self.state_label, self.score = predict(self.model, image)
		self.state['text'] = CiWork5[self.state_label] + ' (' + str(self.score) + ')'
		if self.state_label <= 2:
			self.state['fg'] = 'red'
			self.hint['fg'] = 'red'
			self.hint['text'] = "Please don't be distracted while working!"
		else:
			self.state['fg'] = 'green'
			self.hint['fg'] = 'green'
			self.hint['text'] = "Focus on working...."


def init_model(opt):
	model = Model()
	model = model.cuda() if opt.cuda else model
	return model


def load_model(opt):
	best_model_path = os.path.join(opt.ckpt_dir, 'best_model')
	best_model_state = torch.load(best_model_path)

	model = init_model(opt)
	model.load_state_dict(best_model_state)

	return model


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--ckpt_dir', type=str, default='checkpoints/')
	parser.add_argument('--cuda', action='store_true')
	parser.add_argument('--device', type=int, default=0)

	options = parser.parse_args()

	# Python tkinter GUI
	root = Tk()
	root.geometry("600x500")
	root.title('Home Working State Detection')
	window = MainWindow(root)

	root.mainloop()


if __name__ == '__main__':
	main()