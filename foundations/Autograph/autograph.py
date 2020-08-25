#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf

def linear_layer(x):
    return 3 * x + 2
@tf.function
def simple_nn(x):
    return tf.nn.relu(linear_layer(x))

def simple_function(x):
    return 3*x


# In[ ]:


#internal look at the auto-generated code
print(tf.autograph.to_code(simple_nn.python_function, experimental_optional_features = None))


# In[9]:


import timeit

cell = tf.keras.layers.LSTMCell(100)

@tf.function
def fn(input, state):
    return cell(input, state)

input = tf.zeros([100, 100])
state = [tf.zeros([100, 100])] * 2
#warm up
cell(input, state)
fn(input, state)

graph_time = timeit.timeit(lambda: cell(input, state), number = 100)
auto_graph_time = timeit.timeit(lambda: fn(input, state), number = 100)
print("graph_time: ", graph_time)
print("auto_graph_time", auto_graph_time)

