from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten
from keras.optimizers import SGD
import numpy as np
from PIL import Image
import math

from keras.datasets import mnist

class GAN(object):
    def __init__(self, input_shape=(28,28,1)):
        self.input_shape = input_shape
        self.generator = self.generator_model()
        self.discriminator = self.discriminator_model()
        self.gan = self.build_model(self.generator, self.discriminator)
    
    def generator_model(self):
        model = Sequential()
        model.add(Dense(input_dim=100, units=1024))
        model.add(Activation('tanh'))
        model.add(Dense(128*7*7))
        model.add(BatchNormalization())
        model.add(Activation('tanh'))
        model.add(Reshape((7, 7, 128), input_shape=(128*7*7,)))
        model.add(UpSampling2D(size=(2, 2)))
        model.add(Conv2D(64, (5, 5), padding='same'))
        model.add(Activation('tanh'))
        model.add(UpSampling2D(size=(2, 2)))
        model.add(Conv2D(1, (5, 5), padding='same'))
        model.add(Activation('tanh'))
        return model

    def discriminator_model(self):
        model = Sequential()
        model.add(Conv2D(64, (5, 5), padding='same', input_shape=self.input_shape))
        model.add(Activation('tanh'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(128, (5, 5)))
        model.add(Activation('tanh'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(1024))
        model.add(Activation('tanh'))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        return model

    def build_model(self, g, d):
        model = Sequential()
        model.add(g)
        d.trainable = False
        model.add(d)
        return model

    def train(self, X_train, Y_train, X_test, Y_test, epochs=100, batch_size=32):
        d_optimizer = SGD(lr=0.0005, momentum=0.9, nesterov=True)
        g_optimizer = SGD(lr=0.0005, momentum=0.9, nesterov=True)
        self.generator.compile(loss='binary_crossentropy', optimizer="SGD")
        self.gan.compile(loss='binary_crossentropy', optimizer=g_optimizer)
        self.discriminator.trainable = True
        self.discriminator.compile(loss='binary_crossentropy', optimizer=d_optimizer)
        for epoch in range(1,epochs):
            print("Epoch {}".format(epoch))
            for index in range(int(X_train.shape[0]/batch_size)):
                rand_vector = np.random.uniform(-1, 1, size=(batch_size, 100))
                image_batch = X_train[index*batch_size:(index+1) * batch_size]
                generated_images = self.generator.predict(rand_vector, verbose=0)
                X = np.concatenate((image_batch, generated_images))
                y = [1] * batch_size + [0] * batch_size
                d_loss = self.discriminator.train_on_batch(X, y)
                rand_vector = np.random.uniform(-1, 1, (batch_size,  100))
                self.discriminator.trainable = False
                g_loss = self.gan.train_on_batch(rand_vector, [1] * batch_size)
                self.discriminator.trainable = True
                print("\rbatch {} d loss: {} g loss: {}".format(index, d_loss, g_loss), end="")
            print()
                

if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = (X_train.astype(np.float32) - 127.5)/127.5
    X_train = X_train[:, :, :, None]
    X_test = X_test[:, :, :, None]
    dcgan = GAN()
    dcgan.train(X_train, y_train, X_test, y_test)
