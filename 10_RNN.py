#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 16:57:52 2018

@author: magalidrumare
@Deep learning for text and sequences by François Cholet.  
"""

# Réucurrent Layer in Keras 

# pseudo Code RNN 
state_t=0
for input_t in input_sequences : 
    output_t=f(input_t,state_t)
    state_t=output_t
    
    
# More detailled speudo code for the RNN 
state_t= 0 
for input_t in input_sequences : 
    output_t= activation(dot(W,input_t)+dot(U, state_t)+b)
    state_t=output_t

# Numpy implementation of a simple RNN 
state_t=0
import numpy as np 
timesteps =100 
input_features = 32 
output_features =64

inputs = np.random.random((timesteps, input_features))
states=np.zeros((output_features))

W = np.random.random ((output_features, input_features))
U = np.random.random ((output_features, output_features))
b= np.random.random ((output_features))

successive_output =[]
for input_t in inputs : 
    output_t = np.tanh(np.dot(W,input_t)+np.dot(U,state_t)+b)
    successive_output.append(output_t)
    state_t = output_t
    final_output_sequence = np.concatenate(successive_output, axis=0)
    
# Keras implementation 
    from keras.models import Sequential 
    from keras.layers import Embedding, SimpleRNN
    model = Sequential()
    model.add(Embedding(10000,32)) # 10000 token embbeding dimension =32
    model.add(SimpleRNN(32))
    model.summary()
    
from keras.datasets import imdb
from keras.preprocessing import sequence

max_features = 10000  # number of words to consider as features
maxlen = 500  # cut texts after this number of words (among top max_features most common words)
batch_size = 32

print('Loading data...')
(input_train, y_train), (input_test, y_test) = imdb.load_data(num_words=max_features)
print(len(input_train), 'train sequences')
print(len(input_test), 'test sequences')


# pad_sequence 
# Transform a list of num_samples sequences (lists of scalars) into a 2D Numpy array
# of shape (num_samples, num_timesteps)
#Sequences that are shorter than num_timesteps are padded with value. 
#Sequences longer than num_timesteps are truncated so that it fits the desired length. 

print('Pad sequences (samples x time)')
input_train = sequence.pad_sequences(input_train, maxlen=maxlen)
input_test = sequence.pad_sequences(input_test, maxlen=maxlen)
print('input_train shape:', input_train.shape)
print('input_test shape:', input_test.shape)

from keras.layers import Dense

model = Sequential()
model.add(Embedding(max_features, 32))
model.add(SimpleRNN(32))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(input_train, y_train,
                    epochs=10,
                    batch_size=128,
                    validation_split=0.2)


