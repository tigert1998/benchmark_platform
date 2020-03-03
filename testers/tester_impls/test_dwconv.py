from .test_single_layer import TestSingleLayer

import tensorflow as tf


class TestDwconv(TestSingleLayer):
    def _generate_tf_model(self, sample):
        _, _, input_imsize, cin, cout, _, _, stride, kernel_size = sample
        assert cin == cout

        inputs, nets = self._pad_before_input(
            [[1, input_imsize, input_imsize, cin]])

        net = nets[0]
        net = tf.nn.depthwise_conv2d(
            net,
            filter=tf.get_variable(
                "dwconv_filter", [kernel_size, kernel_size, cin, 1], dtype=tf.float32, trainable=True
            ),
            strides=[1, stride, stride, 1],
            padding='SAME',
            rate=[1, 1],
            name='the_dwconv'
        )

        outputs = self._pad_after_output([net])
        return inputs, outputs
