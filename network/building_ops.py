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


def global_pooling(features):
    return tf.nn.avg_pool(
        features,
        ksize=[1] + features.get_shape().as_list()[1: 3] + [1],
        strides=[1, 1, 1, 1],
        padding='VALID'
    )


def squeeze_and_excitation(features, mid_channels: int):
    """SE layer
    https://github.com/tensorflow/models/blob/89dd9a4e2548e8a5214bd4e564428d01c206a7db/research/slim/nets/mobilenet/conv_blocks.py#L408
    """
    def gating_fn(features): return tf.nn.relu6(features + 3) * 0.16667

    with tf.variable_scope("squeeze_and_excitation"):
        net = global_pooling(features)
        net = tf.keras.layers.Conv2D(
            filters=mid_channels,
            kernel_size=[1, 1],
            strides=[1, 1],
            padding="same",
        )(net)

        net = tf.nn.relu(net)

        net = tf.keras.layers.Conv2D(
            filters=features.get_shape().as_list()[-1],
            kernel_size=[1, 1],
            strides=[1, 1],
            padding="same",
        )(net)

    return gating_fn(net) * features


def mix_conv(features, num_groups: int, stride: int):
    cin = features.get_shape().as_list()[-1]
    assert cin % num_groups == 0

    with tf.variable_scope("mix_conv"):
        groups = []
        for x, i in zip(tf.split(features, num_groups, axis=-1), range(num_groups)):
            with tf.variable_scope("{}".format(i)):
                kernel_size = i * 2 + 3
                groups.append(depthwise_conv(x, stride, kernel_size))

        return tf.concat(groups, axis=-1)
