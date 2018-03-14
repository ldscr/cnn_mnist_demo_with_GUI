from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from PIL import Image,ImageFilter

#Imports
import numpy as np
import tensorflow as tf
import cnn_mnist

#prepare the input data from a image to the input data for predict
def image_prepare(im):
	width=float(im.size[0])
	height=float(im.size[1])
	newImage=Image.new('L',(28,28),(255))
  #resize the input image, let the longer one of width and width to be 20
	if width>height:
		nheight=int(round((20.0/width*height),0))
		if (nheight==0):
			nheight=1
		img=im.resize((20,nheight),Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
		wtop=int(round(((28-nheight)/2),0))
		newImage.paste(img,(4,wtop))
	else:
		nwidth=int(round((20.0/height*width),0))
		if(nwidth==0):
			nwidth=1
		img=im.resize((nwidth,20),Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
		htop=int(round(((28-nwidth)/2),0))
		newImage.paste(img,(htop,4))
	data=list(newImage.getdata())	
  #get the datas of the image, and change the datas to float32
	tva=[(255-x)*1.0/255.0 for x in data]
	final_data=[np.float32(x) for x in tva]
	return [final_data]
   
#get the predict result
def Predict(img):
  mnist_classifier=tf.estimator.Estimator(model_fn=cnn_mnist.cnn_model_fn,model_dir="/home/model_data")
  x_value=np.asarray(image_prepare(img))
  predict_input_fn=tf.estimator.inputs.numpy_input_fn(x={"x":x_value},batch_size=1,shuffle=False)
  predict_result=mnist_classifier.predict(input_fn=predict_input_fn,predict_keys="classes",checkpoint_path="./model_data/model.ckpt-20000")
  return list(predict_result)
if __name__ == "__main__":
  tf.app.run()
