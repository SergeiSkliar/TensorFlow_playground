# -*- coding: utf-8 -*-
"""cifar10

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1i_CDcF0WWNt3wnNwYnPfCitm4PnEQ6zx
"""

import tensorflow as tf
from tensorflow.keras import datasets, layers, models, optimizers, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np


NUM_CLASSES = 10
EPOCHS = 50
OPTIM = tf.keras.optimizers.RMSprop()


def load_data():
  (X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()

  X_train = X_train.astype('float32')
  X_test = X_test.astype('float32')

#normalize
  mean = np.mean(X_train, axis = (0,1,2,3))
  std = np.std(X_train, axis = (0,1,2,3))
  X_train = (X_train - mean) / (std + 1e-7)
  X_test = (X_test - mean) / (std + 1e-7)

  y_train = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
  y_test = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)

  return X_train, y_train, X_test, y_test

def build():
  model = models.Sequential()
  
  #first block: Conv + Conv + MaxPool + Dropout
  model.add(layers.Conv2D(32, (3,3), padding = 'same', activation='relu', input_shape = X_train.shape[1:]))
  model.add(layers.BatchNormalization())
  model.add(layers.Conv2D(32, (3,3), padding = 'same', activation='relu'))
  model.add(layers.BatchNormalization())
  model.add(layers.MaxPool2D(pool_size=(2,2)))
  model.add(layers.Dropout(0.2))

  #second block: Conv + Conv + MaxPool + Dropout
  model.add(layers.Conv2D(64, (3,3), padding = 'same', activation='relu'))
  model.add(layers.BatchNormalization())
  model.add(layers.Conv2D(64, (3,3), padding = 'same', activation='relu'))
  model.add(layers.BatchNormalization())
  model.add(layers.MaxPool2D(pool_size=(2,2)))
  model.add(layers.Dropout(0.3))


  #third block: Conv + Conv + MaxPool + Dropout
  model.add(layers.Conv2D(128, (3,3), padding = 'same', activation='relu'))
  model.add(layers.BatchNormalization())
  model.add(layers.Conv2D(128, (3,3), padding = 'same', activation='relu'))
  model.add(layers.BatchNormalization())
  model.add(layers.MaxPool2D(pool_size=(2,2)))
  model.add(layers.Dropout(0.4))

  #dense
  model.add(layers.Flatten())
  model.add(layers.Dense(NUM_CLASSES, activation='softmax'))

  return model
  model.summary()

(X_train, y_train, X_test, y_test) = load_data()

model = build()

model.compile(loss = 'categorical_crossentropy', optimizer = OPTIM, metrics = ['accuracy'])


#train
BATCH_SIZE_FIT = 64
BATCH_SIZE_EVAL = 128
model.fit(X_train, y_train, batch_size = BATCH_SIZE_FIT, epochs = EPOCHS, validation_data = (X_test, y_test))

score = model.evaluate(X_test, y_test, batch_size = BATCH_SIZE_EVAL)

print("\nTest score: ", score[0])
print("\nTest accuracy: ", score[1])

