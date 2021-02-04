# %%
import tensorflow_hub as hub

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

# Load BigGAN-deep 128 module.
module = hub.Module('https://tfhub.dev/deepmind/biggan-deep-128/1')

# Sample random noise (z) and ImageNet label (y) inputs.
batch_size = 1
truncation = 0.5 # scalar truncation value in [0.0, 1.0]
z = truncation * tf.random.truncated_normal([batch_size, 128]) # noise sample
y_index = tf.random.uniform([batch_size], maxval=1000, dtype=tf.int32)
y = tf.one_hot(y_index, 1000) # one-hot ImageNet label

# Call BigGAN on a dict of the inputs to generate a batch of images with shape
# [8, 128, 128, 3] and range [-1, 1].
samples = module(dict(y=y, z=z, truncation=truncation))
print(samples)

#%%
import numpy as np
import matplotlib.pyplot as plt

init_op = tf.initialize_all_variables()

#run the graph
with tf.Session() as sess:
    sess.run(init_op) #execute init_op
    #print the random values that we sample
    img = sess.run(samples)
img = np.squeeze(img)
plt.imshow(img)
plt.show()

# %%
from PIL import Image
from time import time 

print(img.shape)

rescaled = (255.0 / img.max() * (img - img.min())).astype(np.uint8)

im = Image.fromarray(rescaled)
im.save(f'images/{time()}.png')
# %%
