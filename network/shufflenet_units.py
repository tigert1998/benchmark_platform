import tensorflow as tf

from .building_ops import depthwise_conv, channel_shuffle, grouped_conv, batch_normalization


# https://github.com/megvii-model/ShuffleNet-Series/tree/master/ShuffleNetV1
def shufflenet_v1_unit(features, num_groups: int, mid_channels: int,
                       stride: int, kernel_size: int, num_outputs: int):
    assert stride in [1, 2]

    cin = features.get_shape().as_list()[-1]

    if stride == 1:
        assert num_outputs == cin
        major_branch_num_outputs = num_outputs
    else:
        assert num_outputs > cin
        major_branch_num_outputs = num_outputs - cin

    with tf.variable_scope("shufflenet_v1_unit"):
        net = grouped_conv(features, num_groups, 1, 1, mid_channels)
        net = tf.nn.relu(batch_normalization(net))
        net = channel_shuffle(net, num_groups)
        net = depthwise_conv(net, stride, kernel_size)
        net = batch_normalization(net)
        net = grouped_conv(net, num_groups, 1, 1, major_branch_num_outputs)
        net = batch_normalization(net)

        if stride == 1:
            net = tf.math.add(net, features)
        else:
            minor_branch = tf.keras.layers.AveragePooling2D(
                pool_size=(3, 3), strides=2, padding='same')(features)
            net = tf.concat([minor_branch, net], axis=3)

        net = tf.nn.relu(net)
        return net


def shufflenet_v2_unit(features, stride: int, kernel_size: int):
    cin = features.get_shape().as_list()[-1]

    assert stride in [1, 2]

    with tf.variable_scope("shufflenet_v2_unit"):
        if stride == 1:
            assert cin % 2 == 0
            minor_branch, major_branch = tf.split(features, 2, axis=3)
        else:
            minor_branch = features
            major_branch = features

        major_branch = tf.keras.layers.Conv2D(
            filters=major_branch.get_shape().as_list()[-1],
            kernel_size=[1, 1],
            strides=[1, 1],
            padding='same')(major_branch)
        major_branch = tf.nn.relu(batch_normalization(major_branch))
        major_branch = depthwise_conv(major_branch, stride, kernel_size)
        major_branch = batch_normalization(major_branch)
        major_branch = tf.keras.layers.Conv2D(
            filters=major_branch.get_shape().as_list()[-1],
            kernel_size=[1, 1],
            strides=[1, 1],
            padding='same')(major_branch)
        major_branch = tf.nn.relu(batch_normalization(major_branch))

        if stride == 2:
            minor_branch = depthwise_conv(minor_branch, 2, kernel_size)
            minor_branch = batch_normalization(minor_branch)
            minor_branch = tf.keras.layers.Conv2D(
                filters=minor_branch.get_shape().as_list()[-1],
                kernel_size=[1, 1],
                strides=[1, 1],
                padding='same')(minor_branch)
            minor_branch = tf.nn.relu(batch_normalization(minor_branch))

        net = tf.concat([minor_branch, major_branch], axis=3)
        net = channel_shuffle(net, 2)

        return net
