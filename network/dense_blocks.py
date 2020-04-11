from .building_ops import batch_normalization

import tensorflow as tf


def _conv_block(features, use_bottleneck: bool, kernel_size: int, num_outputs: int):
    with tf.variable_scope("conv_block"):
        net = tf.nn.relu(batch_normalization(features))

        if use_bottleneck:
            net = tf.keras.layers.Conv2D(
                filters=num_outputs * 4,
                kernel_size=[1, 1],
                strides=[1, 1],
                padding='same')(net)
            net = tf.nn.relu(batch_normalization(net))

        net = tf.keras.layers.Conv2D(
            filters=num_outputs,
            kernel_size=[kernel_size, kernel_size],
            strides=[1, 1],
            padding='same')(net)

    return net


# https://github.com/keras-team/keras-applications/blob/master/keras_applications/densenet.py
def dense_block(features, num_layers: int, use_bottleneck: bool, kernel_size: int, growth_rate: int):
    with tf.variable_scope("dense_block"):
        concatenated = features

        for i in range(num_layers):
            with tf.variable_scope("{}".format(i)):
                net = _conv_block(concatenated, use_bottleneck,
                                  kernel_size, growth_rate)
                concatenated = tf.concat([concatenated, net], axis=3)

    return concatenated
