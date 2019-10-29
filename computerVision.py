import tensorflow as tf 
from tensorflow import keras
import numpy as np 
#import matplotlib.pyplot as plt

#fashion mnist dataset for training and testing:
mnist = keras.datasets.fashion_mnist


#loading training and test data
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

#checking training data (ideally dont include this in actual working code -- SK)
#plt.imshow(training_images[0])
print(training_labels[0])
print(training_images[0])

#we write the following line to "normalize" values (which are in a range of 0-255 to instead be 0-1):
training_images = training_images/255.0
test_images = test_images/255.0

#MODEL:
model = keras.Sequential([
	keras.layers.Flatten(input_shape = (28, 28)),
	keras.layers.Dense(128, activation = tf.nn.relu),
	keras.layers.Dense(10, activation = tf.nn.softmax)
	])

#Sequential: That defines a SEQUENCE of layers in the neural network

#Flatten: Flatten just takes that square and turns it into a 1 dimensional set.

#Dense: Adds a layer of neurons

#Each layer of neurons need an activation function to tell them what to do. 

#Relu effectively means "If X>0 return X, else return 0" -- so what it does it it only passes values 0 or greater to the next layer in the network.

#Softmax takes a set of values, and effectively picks the biggest one, so, for example, if the output of the last layer looks like [0.1, 0.1, 0.05, 0.1, 9.5, 0.1, 0.05, 0.05, 0.05], it saves you from fishing through it looking for the biggest value, and turns it into [0,0,0,0,1,0,0,0,0] 

model.compile(optimizer = tf.train.AdamOptimizer(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])
#TRAIN
model.fit(training_images, training_labels, epochs=5)

#TEST
model.evaluate(test_images, test_labels)