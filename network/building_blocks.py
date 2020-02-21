import tensorflow as tf


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
            filter=tf.get_variable(
                "kernel",
                [kernel_size, kernel_size, cin, 1],
                dtype=features.dtype, trainable=True
            ),
            strides=[1, stride, stride, 1],
            padding='SAME',
            rate=[1, 1],
        )


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
                   stride: int, kernel_size: int, num_outputs: int):
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

        net = tf.nn.relu6(depthwise_conv(features, stride, kernel_size))

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


def batch_normalization(features):
    # prevent bias from removal
    return tf.layers.batch_normalization(
        features,
        beta_initializer=tf.constant_initializer(1e-3)
    )


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
            net = net + features
        else:
            minor_branch = tf.keras.layers.AveragePooling2D(
                pool_size=(3, 3), strides=2, padding='same')(features)
            net = tf.concat([minor_branch, net], axis=-1)

        net = tf.nn.relu(net)
        return net
