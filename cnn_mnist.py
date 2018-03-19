from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#Imports
import numpy as np
import tensorflow as tf

#set_verbosity set the threshold for what massage will be logged
#here it means save the INFO massage into the log file
tf.logging.set_verbosity(tf.logging.INFO)

#now the network logic

'''Model Fouction for CNN'''
def cnn_model_fn(features,labels,mode):
  #Input Layer
  input_layer=tf.reshape(features["x"],[-1,28,28,1])

  #Convolutional Layer #1
  conv1=tf.layers.conv2d(
    inputs=input_layer,
    filters=32,
    kernel_size=5,
    #padding instructs TensorFlow to add 0 values to the edges of the input tensor or not.
    #only have two values, default value is valid. and the other one is same. if the value is same
    #then TensorFlow will add 0 to the input tensor and this will lead the output tensor has the same size of the input
    #(for example, without padding, a 5x5 filter over a 28x28 tensor will produce a 24x24 tensor. and if set it to be same, then the output is 28x28.
    padding="same",
    activation=tf.nn.relu)

  #Pooling Layer #1
  pool1=tf.layers.max_pooling2d(inputs=conv1,pool_size=[2,2],strides=2)

  #Convolutional Layer #2 and Pooling layer #2
  conv2=tf.layers.conv2d(
    inputs=pool1,
    filters=64,
    kernel_size=[5,5],
    padding="same",
    activation=tf.nn.relu)
  pool2=tf.layers.max_pooling2d(inputs=conv2,pool_size=2,strides=2)

  #Dense Layer
  #which means full connected layer. and before this, need to flatten each feature maps for one image.
  #here every sample image has 64 feature maps,and every feature maps is 7*7 size.
  pool2_flat=tf.reshape(pool2,[-1,7*7*64])
  dense=tf.layers.dense(inputs=pool2_flat,units=1024,activation=tf.nn.relu)
  #when over the full connected layer, need to go through the dropout way, it indicates that the elements will be randomly dropout during training at the rate you set
  #the training argument get a boolean results of the mode judge. if the mode is train, then perform the dropout op.
  dropout=tf.layers.dropout(inputs=dense,rate=0.4,training=mode==tf.estimator.ModeKeys.TRAIN)

  #Logits Layer
  logits=tf.layers.dense(inputs=dropout,units=10)
  #need prediction argument when in PREDICT and EVALUATE mode
  predictions={
    #return the #1 axis index of the max value
    "classes":tf.argmax(input=logits,axis=1),
    #show the probabilities of every class, and name this operation as softmax_tensor
    "probabilities":tf.nn.softmax(logits,name="softmax_tensor")
    }
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode,predictions=predictions)
  #Calculate Loss(for both Train and Evaluate mode)
  #before calculate the loss, need to change the labels to one_hot format
  onehot_labels=tf.one_hot(indices=tf.cast(labels,tf.int32),depth=10)
  #when use the entropy function below, it will perform softmax activation on logits, and then compute cross entropy with the onehot_labels
  #loss=tf.losses.sparse_softmax_cross_entropy(labels=labels,logits=logits)
  loss=tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels,logits=logits)
  #configure the Training OP
  #if the mode is train, then should optimize the metrics
  #Configure the Training Op(for Train mode)
  if mode==tf.estimator.ModeKeys.TRAIN:
    optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op=optimizer.minimize(loss=loss,global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode,loss=loss,train_op=train_op)
  #Add Evaluation metrics
  eval_metric_ops={
    "accuracy":tf.metrics.accuracy(labels=labels,predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(mode=mode,loss=loss,eval_metric_ops=eval_metric_ops)
def main(unused_argv):
  #load training and evaluation data
  mnist=tf.contrib.learn.datasets.load_dataset("mnist")
  train_data=mnist.train.images
  train_labels=np.asarray(mnist.train.labels,dtype=np.int32)
  eval_data=mnist.test.images
  eval_labels=np.asarray(mnist.test.labels,dtype=np.int32)
  #Create the Estimator
  mnist_classifier=tf.estimator.Estimator(model_fn=cnn_model_fn,model_dir="/home/model_data")
  #Set up logging for predictions
  tensors_to_log={"probabilities":"softmax_tensor"}
  logging_hook=tf.train.LoggingTensorHook(tensors=tensors_to_log,every_n_iter=5000)
  #Trian the Model
  train_input_fn=tf.estimator.inputs.numpy_input_fn(
    x={"x":train_data},
    y=train_labels,
    batch_size=100,
    num_epochs=None,
    shuffle=True)
  mnist_classifier.train(
    input_fn=train_input_fn,
    steps=20000,
    hooks=[logging_hook])
  #Evaluate the model and print the results
  eval_input_fn=tf.estimator.inputs.numpy_input_fn(
    x={"x":eval_data},
    y=eval_labels,
    num_epochs=1,
    shuffle=False)
  eval_results=mnist_classifier.evaluate(input_fn=eval_input_fn)
  print(eval_results)
if __name__ == "__main__":
  tf.app.run()
