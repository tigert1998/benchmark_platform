import tensorflow as tf


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


def mbnet_v1_block(features, stride: int, kernel_size: int, num_outputs: int):
    cin = features.get_shape().as_list()[-1]

    with tf.variable_scope("mbnet_v1_block"):
        net = tf.nn.relu6(tf.nn.depthwise_conv2d(
            features,
            filter=tf.get_variable(
                "weight", [kernel_size, kernel_size, cin, 1],
                dtype=features.dtype, trainable=True
            ),
            strides=[1, stride, stride, 1],
            padding='SAME',
            rate=[1, 1],
        ))
        net = tf.nn.relu6(tf.keras.layers.Conv2D(
            filters=num_outputs,
            kernel_size=[1, 1],
            strides=[1, 1],
            padding='same')(net))

    return net


def mbnet_v2_block(features, expansion_rate: int,
                   stride: int, kernel_size: int, num_outputs: int):
    cin = features.get_shape().as_list()[-1]
    cout = num_outputs
    if stride == 1:
        assert cin == cout

    with tf.variable_scope("mbnet_v2_block"):
        with tf.variable_scope("first_conv"):
            net = tf.nn.relu6(tf.keras.layers.Conv2D(
                filters=cin * expansion_rate,
                kernel_size=[1, 1],
                strides=[1, 1],
                padding='same')(features))

        with tf.variable_scope("depthwise_conv"):
            net = tf.nn.relu6(tf.nn.depthwise_conv2d(
                net,
                filter=tf.get_variable(
                    "kernel",
                    [kernel_size, kernel_size, cin * expansion_rate, 1],
                    dtype=features.dtype, trainable=True
                ),
                strides=[1, stride, stride, 1],
                padding='SAME',
                rate=[1, 1],
            ))

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
