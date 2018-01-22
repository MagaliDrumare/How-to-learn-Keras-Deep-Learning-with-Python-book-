#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 08:33:14 2018

@author: magalidrumare
@ copyright https://github.com/fchollet/deep-learning-with-python-notebooks
"""

# Two class classification  : classify movie reviews into "positive" review or "negative" review 
# based on the text content of the reviews 
# IMDB Dataset" : 25000 training reviews and 25000 testing reviews (50% positive, 50% negative)


# Import pre-processing data with Keras 
from keras.datasets import imdb
# num_words=10000 only keep the 10000 most frequently word in the training data
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
# for train_data and test_data each data is an encoding sequence of words
train_data[0]
# 0 = negative ; 1 = positive 
train_labels[0]


# Preparing the Data : 

# Prepare the inputs data : one-hot encoding 
#Tranform the list of words into tensors
# Hot Encode our list to turn them into vertors of 0 and 1. 
# ->[3,4] => 10000 vectors with zeros except for indices 3 and 5. 
import numpy as np

def vectorize_sequences(sequences, dimension=10000):
    # Create an all-zero matrix of shape (len(sequences), dimension)
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.  # set specific indices of results[i] to 1s
    return results

# Our vectorized training data
x_train = vectorize_sequences(train_data)
# Our vectorized test data
x_test = vectorize_sequences(test_data)

#Prepare the labels : convertion to an array of type float32
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')


# Build the Network : add.(layers.Dense())
# Input (one_hot encoding text)
#->Sequential ([Dense, units=16]->[Dense, units=16]->[Dense, units=1])
#-> Output a probabilty to be 1 
from keras import models 
from keras import layers 

network=models.Sequential()
network.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
network.add(layers.Dense(16, activation='relu'))
network.add(layers.Dense(1, activation='sigmoid'))

# Compilation Step : Loss function, Optimizer, Metrics : network.compile 
# Crossentropy measures the distance between the true prediction and the predictions of the network
# network.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

from keras import optimizers
from keras import losses 
from keras import metrics 

network.compile(optimizer=optimizers.RMSprop(lr=0.001), 
                loss=losses.binary_crossentropy, 
                metrics=[metrics.binary_accuracy])


# Validating your approach on a validation set 
x_val=x_train[:10000]
partial_x_train=x_train[10000:]

y_val=y_train[:10000]
partial_y_train=y_train[10000:]

# Training the network : network.fit
history=network.fit(partial_x_train,
            partial_y_train,
            epochs=20,
            batch_size=512, 
            validation_data=(x_val,y_val))

# acc,val_acc,loss,val_loss 
history_dict = history.history
history_dict.keys()

acc = history.history['binary_accuracy']
val_acc = history.history['val_binary_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

# Loss Representation : 
import matplotlib.pyplot as plt
# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.clf()   # clear figure

# Accuracy Reprensatation 
acc_values = history_dict['binary_accuracy']
val_acc_values = history_dict['val_binary_accuracy']
plt.plot(epochs, acc_values , 'bo', label='Training acc')
plt.plot(epochs,val_acc_values , 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
# => Overfitting 


# Train the model on the training set and evaluate the model on the test data 
#model.add
model=models.Sequential()
model.add(layers.Dense(16,activation='relu',input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
# model.complie
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics= ['accuracy'])
# model.fit 
model.fit(x_train,y_train,epochs=4, batch_size=512)

# model.evaluate 
results=model.evaluate(x_test,y_test)



















