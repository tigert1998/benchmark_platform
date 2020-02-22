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


def dense_block(features, num_layers: int, use_bottleneck: bool, kernel_size: int, growth_rate: int):
    with tf.variable_scope("dense_block"):
        previous_layers = [features]

        for i in range(num_layers):
            with tf.variable_scope("{}".format(i)):
                if len(previous_layers) == 1:
                    concatenated = previous_layers[0]
                else:
                    concatenated = tf.concat(
                        [concatenated, previous_layers[-1]], axis=-1)
                net = _conv_block(concatenated, use_bottleneck,
                                  kernel_size, growth_rate)
                previous_layers.append(net)

    return net
