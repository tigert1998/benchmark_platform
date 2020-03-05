from .test_single_layer import TestSingleLayer

import tensorflow as tf


class TestConv(TestSingleLayer):
    def _generate_tf_model(self, sample):
        _, _, input_imsize, cin, cout, _, _, stride, kernel_size = sample

        inputs, nets = self._pad_before_input(
            [[1, input_imsize, input_imsize, cin]])

        net = nets[0]
        net = tf.keras.layers.Conv2D(
            filters=cout,
            kernel_size=[kernel_size, kernel_size],
            strides=[stride, stride],
            padding='same', name="the_conv"
        )(net)

        outputs = self._pad_after_output([net])
        return inputs, outputs
