import tensorflow as tf
import numpy
from tensorflow import keras

#Network and training parameters
EPOCHS = 200
BATCH_SIZE = 128
VERBOSE = 1
NB_CLASSES = 10 #number of outputs = number of digits
N_HIDDEN = 128
VALIDATION_SPLIT = 0.2 #save for validation

#load mnist dataset
mnist = keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

RESHAPED = 784 #X_train is 60000 rows of 28x28 values; reshape it to 60000x784
X_train = X_train.reshape(60000, RESHAPED)
X_test = X_test.reshape(10000, RESHAPED)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

#Input normalization to be within [0, 1]
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train_samples')
print(X_test.shape[0], 'test_samples')

#One-hot representation of the labels
y_train = tf.keras.utils.to_categorical(y_train, NB_CLASSES)
y_test = tf.keras.utils.to_categorical(y_test, NB_CLASSES)


#Build the model
model = tf.keras.models.Sequential()
model.add(keras.layers.Dense(NB_CLASSES, input_shape = (RESHAPED,), name = 'dense_layer', activation = 'softmax'))


#Compiling the model
model.compile(optimizer = 'SGD', loss = 'categorical_crossentropy', metrics = ['accuracy'])


#Training the model
model.fit(X_train, y_train, batch_size = BATCH_SIZE, epochs = EPOCHS, verbose = VERBOSE, validation_split = VALIDATION_SPLIT)


#Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print("Test accuracy: ", test_acc)
