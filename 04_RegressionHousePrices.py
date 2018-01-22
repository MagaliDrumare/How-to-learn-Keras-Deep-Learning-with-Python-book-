#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 13:25:55 2018

@author: magalidrumare
copyright : https://github.com/fchollet/deep-learning-with-python-notebooks
"""

# Regression : Predict Boston House prices.  
# Boston_Housing Dataset: 506 records (404 training samples, 102 tests-samples). 

# Import the data 
from keras.datasets import boston_housing
(train_data, train_targets),(test_data, test_targets)= boston_housing.load_data()

train_data.shape
test_data.shape 

# The target are the median values of owner occupied home in thousands of dollars
# between 10.000 and 50.000§
train_targets


# Normalization of the train_data ansd test_data 
# Data have differents ranges-> Need to normalize the inputs 
# Substratct the mean and devide by the standard deviation 

mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std
# Quantities we use for normalizing the test_data, using the train_data 
test_data -= mean
test_data /= std

# Building the network function def build_model():
#train_data.shape[1] =13 as -> train_data.shape = (404, 13)
from keras import models
from keras import layers

def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64,activation='relu', input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    # no need of actiovation function. layer is purely linear
    # the network is free to predict values in any range 
    model.add(layers.Dense(1))
    # mse : mean squared error and mae : mean absolute error 
    # mae of 0.5 is 500 & on average. 
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model 


# K_Fold Validation 
# If little data available, validation and test sets contain 
# Too few examples to be statically representative.
# Splitting the data available into K partitions (K=4 or K=5)
# Training K-1 models while evaluating on the remain partition. 
 
import numpy as np

k = 4
num_val_samples = len(train_data) // k
num_epochs = 100
all_scores = []

for i in range(k):
    
    # Prepare the validation data: data from partition # k
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]

    # Prepare the training data: data from all other partitions
    partial_train_data = np.concatenate(
        [train_data[:i * num_val_samples],
         train_data[(i + 1) * num_val_samples:]],
        axis=0)
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples],
         train_targets[(i + 1) * num_val_samples:]],
        axis=0)
        
# Build the Keras model (already compiled)
    model = build_model()
# Train the model (in silent mode, verbose=0)
    model.fit(partial_train_data, partial_train_targets,
              epochs=num_epochs, batch_size=1, verbose=0)
    
# Evaluate the model on the validation data
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    all_scores.append(val_mae)    

all_scores
np.mean(all_scores)


# Train it on the entirety of the data.
model.fit(train_data, train_targets,
          epochs=80, batch_size=16, verbose=0)
test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)
test_mae_score















