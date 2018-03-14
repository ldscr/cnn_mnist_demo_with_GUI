# cnn_mnist_demo_with_GUI
handwriting on a GUI, and predict the digit be wrote.

first build the cnn--->train it with MNIST dataset to get the model--->write a python program reading a img and useing the trained model to predict---->write a simple GUI to use the python program created by the fommer one.

first build the cnn---->cnn_mnist.py 
                    followed the tutorial of the tensorflow website https://www.tensorflow.org/tutorials/layers 

get the model------->saved in the directory model_data
                    if don't want to waste time trainning the network. can use the trained model directly
read img and use trained model to predict------->cnn_mnist_predict_with_img.py
the GUI -------->cnn_mnist_predict_with_GUI.py

the environment:
    lsb-release -a : Ubuntu 16.04 (64 bit)
    nvidia-smi: Driver Version:390.25
    docker:   1.0.1
    tensorflow container: 1.5.0
     
