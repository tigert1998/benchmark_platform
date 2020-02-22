import tensorflow as tf

from .building_ops import depthwise_conv, squeeze_and_excitation

from typing import Optional


def mbnet_v1_block(features, stride: int, kernel_size: int, num_outputs: int):
    cin = features.get_shape().as_list()[-1]

    assert stride in [1, 2]

    with tf.variable_scope("mbnet_v1_block"):
        net = tf.nn.relu6(depthwise_conv(features, stride, kernel_size))
        net = tf.nn.relu6(tf.keras.layers.Conv2D(
            filters=num_outputs,
            kernel_size=[1, 1],
            strides=[1, 1],
            padding='same')(net))

    return net


def mbnet_v2_block(features, expansion_rate: int,
                   stride: int, kernel_size: int, num_outputs: int,
                   se_mid_channels: Optional[int] = None):
    cin = features.get_shape().as_list()[-1]
    cout = num_outputs
    assert stride in [1, 2]
    if stride == 1:
        assert cin == cout

    with tf.variable_scope("mbnet_v2_block"):
        with tf.variable_scope("first_conv"):
            net = tf.nn.relu6(tf.keras.layers.Conv2D(
                filters=cin * expansion_rate,
                kernel_size=[1, 1],
                strides=[1, 1],
                padding='same')(features))

        net = tf.nn.relu6(depthwise_conv(net, stride, kernel_size))
        if se_mid_channels is not None:
            net = squeeze_and_excitation(net, se_mid_channels)

        with tf.variable_scope("second_conv"):
            net = tf.keras.layers.Conv2D(
                filters=num_outputs,
                kernel_size=[1, 1],
                strides=[1, 1],
                padding='same'
            )(net)

        if stride == 1:
            net = features + net

    return net
