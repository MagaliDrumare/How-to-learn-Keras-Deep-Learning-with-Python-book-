#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 15:14:14 2018

@author: magalidrumare
@ copyright https://github.com/fchollet/deep-learning-with-python-notebooks
"""

# Regularizations techniques on the IBDM Dataset 

# Fundamental issue  in machine learning the tension between optimization and generalization 
# Optimization = best porformance possible on the training data 
# Generalization = best performance on data the model has never seen before. 
# low loss on training data, low loss on tests data -> underfitting. (at the beggining) 
# loss on test data degradating -> overfitting. The model learn patterns that are specific to the training data. 
# To prevent overfitting : get more training data. 
# Regularization -> Force the network to memorize a small number of parterns ans focus on the most prominents paterns. 

# The most common way to prevent over-fitting in Neural networks; 
#-> getting more training data 
#-> reducing the capacity of the network 
#-> adding weight regularization 
#-> adding dropout 

# Following code for regularization weight and dropout. 
#->Option 3 : l2_model.add(layers.Dense(16, kernel_regularizer=regularizers.l2(0.001), activation='relu', input_shape=(10000,)))
#->Option 4 : 



# Training the orginal model 
import keras

from keras.datasets import imdb
import numpy as np

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

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
# Our vectorized labels
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')


from keras import models
from keras import layers

original_model = models.Sequential()
original_model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
original_model.add(layers.Dense(16, activation='relu'))
original_model.add(layers.Dense(1, activation='sigmoid'))

original_model.compile(optimizer='rmsprop',
                       loss='binary_crossentropy',
                       metrics=['acc'])

original_hist = original_model.fit(x_train, y_train,
                                   epochs=20,
                                   batch_size=512,
                                   validation_data=(x_test, y_test))

epochs = range(1, 21)
original_val_loss = original_hist.history['val_loss']

# Option 3 - Adding weight regularization :
# Put contraints on the the complexity of a network by forcing its weights to only take small values 
# ...which make the distrubution of the weight values more regular. 
# Weight regularization and done by adding the loss function a cost associated 
# ...with having large weights. 
# L1 regularization -> cost added = absolute value of the weight coefficients 
# L2 regularization -> cost added = square of the value of the weights coefficients(weight decay)
 
# Add to the first and second layers kernel_regularizers.l2(0.001)
from keras import regularizers

l2_model = models.Sequential()
l2_model.add(layers.Dense(16, kernel_regularizer=regularizers.l2(0.001),
                          activation='relu', input_shape=(10000,)))
l2_model.add(layers.Dense(16, kernel_regularizer=regularizers.l2(0.001),
                          activation='relu'))
l2_model.add(layers.Dense(1, activation='sigmoid'))

l2_model.compile(optimizer='rmsprop',
                 loss='binary_crossentropy',
                 metrics=['acc'])

l2_model_hist = l2_model.fit(x_train, y_train,
                             epochs=20,
                             batch_size=512,
                             validation_data=(x_test, y_test))

l2_model_val_loss = l2_model_hist.history['val_loss']

# Visualization of the impact of the L2Regularization 
import matplotlib.pyplot as plt
plt.plot(epochs, original_val_loss, 'b+', label='Original model')
plt.plot(epochs, l2_model_val_loss, 'bo', label='L2-regularized model')
plt.xlabel('Epochs')
plt.ylabel('Validation loss')
plt.legend()

plt.show()

# Option 4 - Dropout 
# Introducing noise in the output values of a layer can break up patterns that are not significant.
# Randomly dropping out a number of output features of layer durring the training. 
# Output of a layer [0.2,0.5,1.3,1.1]-> after the dropout->[0.2,0,1.3,0]
# No dropout at test time. 


dpt_model = models.Sequential()
dpt_model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
dpt_model.add(layers.Dropout(0.5))
dpt_model.add(layers.Dense(16, activation='relu'))
dpt_model.add(layers.Dropout(0.5))
dpt_model.add(layers.Dense(1, activation='sigmoid'))

dpt_model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['acc'])

dpt_model_hist = dpt_model.fit(x_train, y_train,
                               epochs=20,
                               batch_size=512,
                               validation_data=(x_test, y_test))

dpt_model_val_loss = dpt_model_hist.history['val_loss']

# Visualization of the Dropout  
plt.plot(epochs, original_val_loss, 'b+', label='Original model')
plt.plot(epochs, dpt_model_val_loss, 'bo', label='Dropout-regularized model')
plt.xlabel('Epochs')
plt.ylabel('Validation loss')
plt.legend()

plt.show()



