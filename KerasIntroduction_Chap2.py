#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 17:11:58 2018

@author: magalidrumare


"""

# import the dataset 
from keras.datasets import mnist 

(train_images, train_labels), (test_images, test_labels)=mnist.load_data()

train_images.shape
len(train_labels)
train_labels

test_images.shape
len(test_labels)
test_labels


# Network archirecture
from keras import models 
from keras import layers
network = models.Sequential()
network.add(layers.Dense(512,activation='relu',input_shape=(28*28,)))
network.add(layers.Dense(10, activation='softmax'))

# The compilation steps 
network.compile(optimizer ='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

#preparing the data 
train_images = train_images.reshape((60000,28*28))
train_images = train_images.astype('float32')/255


test_images = test_images.reshape((10000,28*28))
test_images = test_images.astype('float32')/255

# preparing the labels 
from keras.utils import to_categorical 
train_labels=to_categorical(train_labels)
test_labels= to_categorical(test_labels)

# train the network 
network.fit(train_images, train_labels,epochs=5, batch_size=128)











