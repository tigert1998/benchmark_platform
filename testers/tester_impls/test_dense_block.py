from .test_single_layer import TestSingleLayer

import tensorflow as tf

from network.dense_blocks import dense_block


class TestDenseBlock(TestSingleLayer):
    def _generate_tf_model(self, sample):
        _, input_imsize, cin, growth_rate, num_layers, kernel_size = sample

        inputs, nets = self._pad_before_input(
            [[1, input_imsize, input_imsize, cin]])

        net = nets[0]
        net = dense_block(net, num_layers, True, kernel_size, growth_rate)

        outputs = self._pad_after_output([net])
        return inputs, outputs
