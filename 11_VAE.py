#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 16:13:53 2018

@author: magalidrumare
@ copyright François Cholet DeepLearning with Python
"""


# Variational autoencoders 

# -> A classical image autoencoder takes an image, 
# maps it to a latent vector space via an "encoder" module, 
# decode it back to an output with the same dimensions as the original image, 
# via a "decoder" module.

#-> A VAE, instead of compressing its input image into a fixed "code" in the latent space,
# turns the image into the parameters of a statistical distribution:
# a mean and a variance. 
# The VAE then uses the mean and variance parameters to randomly sample one element 
# of the distribution, and decodes that element back to the original input. 

# Part 1 ->Encoding : input_img -> encoder -> z_mean and z_log_variance.
# Part 2 -> Sampling : z = z_mean + exp(z_log_variance) * epsilon / epsilon is a random tensor of small values
# Part 3 -> Decoding : z-> decoder-> reconstructed input_img 
# Loss function of the VAE = Reconstruction loss + Regularization Loss 
# Reconstruction : force the input_img to match reconstructed_img 
# Regularization : learning well-formed latent spaces + reducing overfitting. 




# Part 1 - Encoding 
import keras
from keras import layers
from keras import backend as K
from keras.models import Model
import numpy as np

img_shape = (28, 28, 1)
batch_size = 16
latent_dim = 2  # Dimensionality of the latent space: a plane

# Inputs 
input_img = keras.Input(shape=img_shape)

# Encoder 
x = layers.Conv2D(32, 3,
                  padding='same', activation='relu')(input_img)
x = layers.Conv2D(64, 3,
                  padding='same', activation='relu',
                  strides=(2, 2))(x)
x = layers.Conv2D(64, 3,
                  padding='same', activation='relu')(x)
x = layers.Conv2D(64, 3,
                  padding='same', activation='relu')(x)
shape_before_flattening = K.int_shape(x)

x = layers.Flatten()(x)
x = layers.Dense(32, activation='relu')(x)

# Outputs of the Encoder 
z_mean = layers.Dense(latent_dim)(x)
z_log_var = layers.Dense(latent_dim)(x)




# Part 2- Sampling 
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
                              mean=0., stddev=1.)
    return z_mean + K.exp(z_log_var) * epsilon

z = layers.Lambda(sampling)([z_mean, z_log_var])


# Part 3 - Decoding 
# This is the input where we will feed `z`.
decoder_input = layers.Input(K.int_shape(z)[1:])

# Upsample to the correct number of units
x = layers.Dense(np.prod(shape_before_flattening[1:]),
                 activation='relu')(decoder_input)

# Reshape into an image of the same shape as before our last `Flatten` layer
x = layers.Reshape(shape_before_flattening[1:])(x)

# We then apply then reverse operation to the initial
# stack of convolution layers: a `Conv2DTranspose` layers
# with corresponding parameters.
x = layers.Conv2DTranspose(32, 3,
                           padding='same', activation='relu',
                           strides=(2, 2))(x)

# 1 = 1 image and 3 the channel of the image 
x = layers.Conv2D(1, 3,
                  padding='same', activation='sigmoid')(x)
# We end up with a feature map of the same size as the original input.

# This is our decoder model.
decoder = Model(decoder_input, x)

# We then apply it to `z` to recover the decoded `z`.
z_decoded = decoder(z)




# Part 4-Dual Loss of the VAE 

class CustomVariationalLayer(keras.layers.Layer):

    def vae_loss(self, x, z_decoded):
        x = K.flatten(x)
        z_decoded = K.flatten(z_decoded)
#  Reconstruction : force the input_img to match reconstructed_img                 
        xent_loss = keras.metrics.binary_crossentropy(x, z_decoded)
        
# Regularization : learning well-formed latent spaces + reducing overfitting.         
        kl_loss = -5e-4 * K.mean(
            1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return K.mean(xent_loss + kl_loss)

    def call(self, inputs):
        x = inputs[0]
        z_decoded = inputs[1]
        loss = self.vae_loss(x, z_decoded)
        self.add_loss(loss, inputs=inputs)
        # We don't use this output.
        return x

# We call our custom layer on the input and the decoded output,
# to obtain the final model output.
y = CustomVariationalLayer()([input_img, z_decoded])


# Part 5 - Training the model 
from keras.datasets import mnist

vae = Model(input_img, y)
vae.compile(optimizer='rmsprop', loss=None)
vae.summary()

# Train the VAE on MNIST digits
(x_train, _), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_train = x_train.reshape(x_train.shape + (1,))
x_test = x_test.astype('float32') / 255.
x_test = x_test.reshape(x_test.shape + (1,))

vae.fit(x=x_train, y=None,
        shuffle=True,
        epochs=10,
        batch_size=batch_size,
        validation_data=(x_test, None))