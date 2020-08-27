#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf

def build_model():
    #variable-length sequence of integers
    text_input_a = tf.keras.Input(shape = (None,), dtype = 'int32')
    
    #variable-length sequence of integers
    text_input_b = tf.keras.Input(shape = (None,), dtype = 'int32')
    
    #Embedding for 1000 unique words mapped to 128-dimensional vector
    shared_embedding = tf.keras.layers.Embedding(1000, 128)
    
    #We reuse the same layer to encode both inputs
    encoded_input_a = shared_embedding(text_input_a)
    encoded_input_b = shared_embedding(text_input_b)
    
    #two logistic predictions at the end
    prediction_a = tf.keras.layers.Dense(1, activation = 'sigmoid', name = 'prediction_a')(encoded_input_a)
    prediction_b = tf.keras.layers.Dense(1, activation = 'sigmoid', name = 'prediction_b')(encoded_input_b)
    
    #2 inputs, 2 outputs
    #in the middle - shared model
    model = tf.keras.Model(inputs = [text_input_a, text_input_b], outputs = [prediction_a, prediction_b])
    
    tf.keras.utils.plot_model(model, to_file = 'shared_model.png')
    
build_model()


# In[ ]:




