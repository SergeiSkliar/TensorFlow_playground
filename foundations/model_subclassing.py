#!/usr/bin/env python
# coding: utf-8

# In[18]:


import tensorflow as tf
from tensorflow.keras import layers

class MyLayer(layers.Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        #create a trainable weight variable for this layer
        self.kernel = self.add_weight(name = 'kernel', shape = (input_shape[1], self.output_dim),
                                     initializer = 'uniform', trainable = True)
        
    def call(self, inputs):
        #do the multiplication and return
        return tf.matmul(inputs, self.kernel)


# In[19]:


model = tf.keras.Sequential([MyLayer(20), layers.Activation('softmax')])

