import os
import argparse

import torch

from tkinter import * 
from tkinter import filedialog
from PIL import Image, ImageTk

from model import WorkStateClsModel as Model


class MainWindow(Frame):
	def __init__(self, master):
		Frame.__init__(self, master)
		self.master = master

		self.btn = Button(self, text='Upload Image', fg='black', bg='gray', command=self.pressUpload)
		self.btn.place(x=240, y=420, height=40, width=100)

		self.label = Label(self, text="Current Image's Working State is:  ", font=('Times New Roman', 15, 'bold'))
		self.label.place(x=80, y=350, height=50, width=300)

		self.state = Label(self, text="________________", font=('Times New Roman', 15, 'bold'))
		self.state.place(x=400, y=350, height=50, width=100)

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
		img = Image.open(selectImg).resize((300, 300))
		render = ImageTk.PhotoImage(img)

		self.image = Label(self, image=render)
		self.image.image = render
		self.image.place(x=150, y=30, height=300, width=300)

		# Change State



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

	#model = load_model(options)

	# Python tkinter GUI
	root = Tk()
	root.geometry("600x500")
	root.title('Home Working State Detection')
	window = MainWindow(root)

	root.mainloop()


if __name__ == '__main__':
	main()