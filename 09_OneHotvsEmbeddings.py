#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 15:29:25 2018

@author: magalidrumare
@ copyright  https://github.com/fchollet/deep-learning-with-python-notebooks

"""

# One-hot word vectors versus word embeddings 
# Two ways to associate a vector with a word 
# One-hot word vector is sparsen high-dimensional (20000), hard-coded
# Word embedding is dense, lower dimensional (256,512,1024) learned from data 


# One-hot encoding numbers : to_categorical 
data = [1, 3, 2, 0, 3, 2, 2, 1, 0, 1]
from numpy import array
from numpy import argmax
from keras.utils import to_categorical
# define example
data = [1, 3, 2, 0, 3, 2, 2, 1, 0, 1]
data = array(data)
print(data)
# one hot encode
encoded = to_categorical(data)
print(encoded)
# invert encoding
inverted = argmax(encoded[0])
print(inverted)

# One-hot encodings text : Tokenizer 
from keras.preprocessing.text import Tokenizer 
samples =['The cat sat on the mat', 'The dog ate my homework']
tokenizer=Tokenizer(num_words =1000)
# build the word index 
tokenizer.fit_on_texts(samples)
sequences = tokenizer.texts_to_sequences(samples)
one_hit_results=tokenizer.texts_to_matrix(samples, mode='binary')
word_index=tokenizer.word_index
len(word_index)
    
# Word Embeddings 
# Word Embeddingsvare meant to map human language into a geometric space. 
# To learn the word vectors in the same way you learn the weights of a neural network. 
# Pre-trained word embeddings.
# The embedding layer is a dictionnary mapping integer indices (which stand for specific words) 
# to dense vector : input integers-> output a vector. 
# inputs : 2D tensors integers(samples, sequence_lenght) all sequances must have the same lenght.
# outputs : 3D tensors (samples, sequences_lenght, embeddings_dimensionnality)
# Embedding layers : weights are initially random 

from keras.datasets import imdb
from keras import preprocessing
from keras.layers import Embedding

# Number of words to consider as features
max_features = 10000
# Cut texts after this number of words 
# (among top max_features most common words)
maxlen = 20

# Load the data as lists of integers.
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

# This turns our lists of integers
# into a 2D integer tensor of shape `(samples, maxlen)`
x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

from keras.models import Sequential
from keras.layers import Flatten, Dense

model = Sequential()
# We specify the maximum input length to our Embedding layer
# so we can later flatten the embedded inputs
model.add(Embedding(10000, 8, input_length=maxlen))
# After the Embedding layer, 
# our activations have shape `(samples, maxlen, 8)`.

# We flatten the 3D tensor of embeddings 
# into a 2D tensor of shape `(samples, maxlen * 8)`
model.add(Flatten())

# We add the classifier on top
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
model.summary()

history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_split=0.2)




