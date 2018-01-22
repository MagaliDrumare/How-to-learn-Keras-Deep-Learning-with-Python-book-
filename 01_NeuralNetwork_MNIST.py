#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 07:48:53 2018

@author: magalidrumare
copyright : https://github.com/fchollet/deep-learning-with-python-notebooks
"""

# Classify grayscale image of handwritten digit 28x28 into 10 categories (O to 9)
# MNIST dataset : 60000 training images 10000 test images. 

# Import Keras 
import keras 

# Download the pre-processing dataset 
from keras.datasets import mnist

# Create the training and test 
(train_images, train_labels), (test_images, test_labels)= mnist.load_data()


# Build the network : netword.add
# Core building block of the neural networks is the layer 
# A data-processing module, a filter for data 
from keras import models 
from keras import layers 
network = models.Sequential()
# Two dense layers = fully connected layers 
network.add(layers.Dense(512, activation ='relu', input_shape=(28 * 28,)))
# Activation is a softmax function : output will be 10 probability scrores (summing to one)
# Each score will be the probability that the current digit image belongs to one of the 10 digit classes 
network.add(layers.Dense(10, activation='softmax'))


# Compilation Step : Loss function, Optimizer, Metrics : network.compile 
network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Reshape the data into shape that the networl expects : .reshape, .astype
# Reshape  train and test images -> into a float32 array of shape (6000,28*28) with value [0,1]
train_images= train_images.reshape((60000,28 * 28))
train_images = train_images.astype('float32')/255

test_images = test_images.reshape((10000,28 * 28))
test_images =test_images.astype('float32')/255

#Categorically encode the labels : to categorical 
from keras.utils import to_categorical 
train_labels= to_categorical(train_labels)
test_labels =to_categorical (test_labels)


# Training step : network.fit 
network.fit(train_images, train_labels,epochs=5,batch_size=128)

#Check the performance on the Test Set :network.evaluate
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc:', test_acc)



