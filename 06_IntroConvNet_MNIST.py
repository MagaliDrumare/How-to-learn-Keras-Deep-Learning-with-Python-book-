#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 17:04:16 2018

@author: magalidrumare
@ copyright https://github.com/fchollet/deep-learning-with-python-notebooks
"""

# Use ConvNet to classify MNIST Project 
# convnet takes input tensor og size (image height, image width, image_channels)
# input_shape = (28,28,1) # image_channels for black and white = 1, for colour =3. 

from keras import layers 
from keras import models 

model=models.Sequential()
# Convolultion layer 32 features map, 3x3 kernel, activation function =relu. 
model.add(layers.Conv2D(32,(3,3),activation='relu', input_shape=(28,28,1)))
model.add(layers.MaxPooling2D((2,2)))
# No need to put the Inputshape 
model.add(layers.Conv2D(64,(3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3), activation='relu'))
# 3D tensor converted to 1D tensor 
# (3,3,64 )-> (,576)
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.summary()

# Normalize the images and the labels 
from keras.datasets import mnist
from keras.utils import to_categorical

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Train the model 
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5, batch_size=64)

# Evaluate the model with Test Data 
test_loss, test_acc = model.evaluate(test_images, test_labels)



