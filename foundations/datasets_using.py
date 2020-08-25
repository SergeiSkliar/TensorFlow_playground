#!/usr/bin/env python
# coding: utf-8

# In[3]:


import tensorflow as tf
import tensorflow_datasets as tfds

builders = tfds.list_builders()
print (builders)

data, info = tfds.load("mnist", with_info = True)

train_data, test_data = data['train'], data['test']
print(info)

