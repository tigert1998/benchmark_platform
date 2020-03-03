from .test_single_layer import TestSingleLayer

import tensorflow as tf


class TestDilatedConv(TestSingleLayer):
    def _generate_tf_model(self, sample):
        _, input_imsize, cin, cout, dilation, stride, kernel_size = sample

        inputs, nets = self._pad_before_input(
            [[1, input_imsize, input_imsize, cin]])

        net = nets[0]
        net = tf.keras.layers.Conv2D(
            filters=cout,
            kernel_size=[kernel_size, kernel_size],
            strides=[stride, stride],
            padding='same',
            dilation_rate=dilation,
            name="the_dilated_conv"
        )(net)

        outputs = self._pad_after_output([net])
        return inputs, outputs
