import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.initializers import glorot_normal
from tensorflow.keras.regularizers import l2


def create_dense(size):
    return Dense(units=size, activation=tf.nn.relu,
                 kernel_initializer=glorot_normal,
                 bias_initializer=glorot_normal,
                 kernel_regularizer=l2,
                 bias_regularizer=l2)


def create_output(size):
    return Dense(units=size,
                 activation=tf.nn.softmax,
                 kernel_initializer=glorot_normal,
                 bias_initializer=glorot_normal,
                 kernel_regularizer=l2,
                 bias_regularizer=l2)
