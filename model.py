# Copyright (c) 2016 Matthew Earl
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
#     The above copyright notice and this permission notice shall be included
#     in all copies or substantial portions of the Software.
# 
#     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#     OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
#     MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
#     NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
#     DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
#     OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
#     USE OR OTHER DEALINGS IN THE SOFTWARE.


"""
Definition of the neural networks. 

"""


__all__ = (
    'get_training_model',
    'get_detect_model',
    'WINDOW_SHAPE',
)


import tensorflow as tf
import common


# WINDOW_SHAPE = (64, 128)
WINDOW_SHAPE = (96, 120)


# Utility functions
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def conv2d(x, W, stride=(1, 1), padding='SAME'):
  return tf.nn.conv2d(x, W, strides=[1, stride[0], stride[1], 1],
                      padding=padding)


def max_pool(x, ksize=(2, 2), stride=(2, 2)):
  return tf.nn.max_pool(x, ksize=[1, ksize[0], ksize[1], 1],
                        strides=[1, stride[0], stride[1], 1], padding='SAME')


def avg_pool(x, ksize=(2, 2), stride=(2, 2)):
  return tf.nn.avg_pool(x, ksize=[1, ksize[0], ksize[1], 1],
                        strides=[1, stride[0], stride[1], 1], padding='SAME')


def convolutional_layers():
    x = tf.placeholder(tf.float32, [None, None, None])

    # First layer
    W_conv1 = weight_variable([5, 5, 1, 48])
    b_conv1 = bias_variable([48])
    x_expanded = tf.expand_dims(x, 3)
    h_conv1 = tf.nn.relu(conv2d(x_expanded, W_conv1) + b_conv1)
    h_pool1 = max_pool(h_conv1, ksize=(2, 2), stride=(2, 2))

    # Second layer
    W_conv2 = weight_variable([5, 5, 48, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool(h_conv2, ksize=(2, 1), stride=(2, 1))

    # Third layer
    W_conv3 = weight_variable([5, 5, 64, 128])
    b_conv3 = bias_variable([128])

    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    h_pool3 = max_pool(h_conv3, ksize=(2, 2), stride=(2, 2))

    return x, h_pool3, [W_conv1, b_conv1,
                        W_conv2, b_conv2,
                        W_conv3, b_conv3]


def get_training_model():
    x, conv_layer, conv_vars = convolutional_layers()
    W_fc1 = weight_variable([30 * 12 * 128, 2048])
    b_fc1 = bias_variable([2048])

    conv_layer_flat = tf.reshape(conv_layer, [-1, 30 * 12 * 128])
    h_fc1 = tf.nn.relu(tf.matmul(conv_layer_flat, W_fc1) + b_fc1)

    # Output layer
    W_fc2 = weight_variable([2048, 1 + 9 * len(common.CHARS)])
    b_fc2 = bias_variable([1 + 9 * len(common.CHARS)])

    y = tf.matmul(h_fc1, W_fc2) + b_fc2

    return (x, y, conv_vars + [W_fc1, b_fc1, W_fc2, b_fc2])


def get_detect_model():
    x, conv_layer, conv_vars = convolutional_layers()
    
    # Fourth layer
    W_fc1 = weight_variable([12 * 30 * 128, 2048])
    W_conv1 = tf.reshape(W_fc1, [12, 30, 128, 2048])
    b_fc1 = bias_variable([2048])
    h_conv1 = tf.nn.relu(conv2d(conv_layer, W_conv1,
                                stride=(1, 1), padding="VALID") + b_fc1) 
    # Fifth layer
    W_fc2 = weight_variable([2048, 1 + 9 * len(common.CHARS)])
    W_conv2 = tf.reshape(W_fc2, [1, 1, 2048, 1 + 9 * len(common.CHARS)])
    b_fc2 = bias_variable([1 + 9 * len(common.CHARS)])
    h_conv2 = conv2d(h_conv1, W_conv2) + b_fc2

    return (x, h_conv2, conv_vars + [W_fc1, b_fc1, W_fc2, b_fc2])

