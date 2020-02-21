import tensorflow as tf

import numpy as np


def channel_shuffle(features, num_groups: int):
    _, h, w, c = features.get_shape().as_list()
    assert c % num_groups == 0

    with tf.variable_scope('channel_shuffle'):
        reshaped = tf.reshape(
            features, [-1, w, num_groups, c // num_groups])
        transposed = tf.transpose(reshaped, [0, 1, 3, 2])
        return tf.reshape(transposed, [-1, h, w, c])


def grouped_conv(features, num_groups: int, stride: int, kernel_size: int, num_outputs: int):
    cin = features.get_shape().as_list()[-1]
    cout = num_outputs
    assert cin % num_groups == 0 and cout % num_groups == 0

    with tf.variable_scope("grouped_conv"):
        groups = [
            tf.keras.layers.Conv2D(
                filters=num_outputs // num_groups,
                kernel_size=[kernel_size, kernel_size],
                strides=[stride, stride],
                padding="same",
                name="{}/conv".format(i)
            )(x) for i, x in zip(range(num_groups), tf.split(features, num_groups, axis=-1))
        ]
        net = tf.concat(groups, axis=-1, name="concat")

    return net


def depthwise_conv(features, stride: int, kernel_size: int):
    cin = features.get_shape().as_list()[-1]
    with tf.variable_scope("depthwise_conv"):
        return tf.nn.depthwise_conv2d(
            features,
            filter=tf.Variable(
                initial_value=tf.ones([kernel_size, kernel_size, cin, 1]),
                name="kernel",
                dtype=features.dtype,
                trainable=True
            ),
            strides=[1, stride, stride, 1],
            padding='SAME',
            rate=[1, 1],
        )


def batch_normalization(features):
    # prevent bias from removal
    return tf.layers.batch_normalization(
        features,
        beta_initializer=tf.constant_initializer(1e-3)
    )
