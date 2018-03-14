from Tkinter import *
from PIL import Image,ImageDraw,ImageFilter
import cnn_mnist_predict_with_img as cnn
import numpy as np
class cnn_mnist_predict_with_GUI:
	def __init__(self):
		#build a window
		self.window=Tk()
		self.window.title("HandWriting Digit Recognition")
		self.img=Image.new('L',(280,280),255)
                #used to clear the img overall
		self.bg_img=Image.new('L',(280,280),255)
		self.draw=ImageDraw.Draw(self.img)

		#configure the canvas
		self.canvas=Canvas(self.window,width=280,height=280,bg="white")		
		self.canvas.pack()
		self.canvas.bind("<B1-Motion>",self.paint)
		#build a frame to hold the widgets
		frame=Frame(self.window)
		frame.pack()
		label=Label(frame,text="Predict Result: ")
		self.predictResultLabel=Label(frame,text=" ")

		#create the Predict Button and the Clear Button
		btPredict=Button(frame,text="Predict",command=self.digitPredict)
		btClear=Button(frame,text="Clear",command=self.displayClear)

		#locate the button on the canvas
		label.grid(row=0,column=1)
		self.predictResultLabel.grid(row=0,column=2)
		btPredict.grid(row=1,column=6)
		btClear.grid(row=1,column=7)

		#loop
		self.window.mainloop()
	def digitPredict(self):
		predict_result=cnn.Predict(self.img)
		self.predictResultLabel["text"]=predict_result
	def displayClear(self):
		self.canvas.delete("all")
		self.predictResultLabel["text"]=" "
		self.img.paste(self.bg_img)
	def paint(self,event):
		x1,y1=(event.x-1),(event.y-1)
		x2,y2=(event.x+1),(event.y+1)
		self.canvas.create_oval(x1,y1,x2,y2,fill="black")
		self.draw.ellipse((x1,y1,x2,y2),fill=0)

cnn_mnist_predict_with_GUI()

