from .test_single_layer import TestSingleLayer

import tensorflow as tf


class TestDilatedConv(TestSingleLayer):
    def _generate_tf_model(self, sample):
        _, input_imsize, cin, cout, dilation, stride, kernel_size = sample

        inputs, nets = self._pad_before_input(
            [[1, input_imsize, input_imsize, cin]])

        net = nets[0]

        with tf.variable_scope("dilated_conv"):
            net = tf.nn.conv2d(
                net,
                filters=tf.Variable(
                    initial_value=tf.ones(
                        [kernel_size, kernel_size, cin, cout]
                    ),
                    name="kernel",
                    dtype=net.dtype,
                    trainable=True
                ),
                strides=[stride, stride],
                padding="SAME",
                data_format='NHWC',
                dilations=dilation,
                name="conv"
            )

        outputs = self._pad_after_output([net])
        return inputs, outputs
