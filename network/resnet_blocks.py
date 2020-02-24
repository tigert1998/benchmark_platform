import tensorflow as tf

from typing import List

from .building_ops import batch_normalization


# https://github.com/keras-team/keras-applications/blob/master/keras_applications/resnet50.py
def resnet_v1_block(features, num_filters_list: List[int], stride: int, kernel_size: int):
    num_filters_1, num_filters_2, num_filters_3 = num_filters_list

    assert stride in [1, 2]
    if stride == 1:
        cin = features.get_shape()[-1]
        assert num_filters_3 == cin

    net = tf.keras.layers.Conv2D(
        filters=num_filters_1,
        kernel_size=[1, 1],
        strides=[stride, stride],
        padding='same'
    )(features)

    net = tf.nn.relu(batch_normalization(net))

    net = tf.keras.layers.Conv2D(
        filters=num_filters_2,
        kernel_size=[kernel_size, kernel_size],
        strides=[1, 1],
        padding='same'
    )(net)

    net = tf.nn.relu(batch_normalization(net))

    net = tf.keras.layers.Conv2D(
        filters=num_filters_3,
        kernel_size=[1, 1],
        strides=[1, 1],
        padding='same'
    )(net)

    net = batch_normalization(net)

    if stride == 1:
        shortcut = features
    else:
        shortcut = tf.keras.layers.Conv2D(
            filters=num_filters_3,
            kernel_size=[1, 1],
            strides=[stride, stride],
            padding='same'
        )(features)
        shortcut = batch_normalization(shortcut)

    return tf.nn.relu(net + shortcut)
