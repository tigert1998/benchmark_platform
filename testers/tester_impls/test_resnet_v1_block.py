from .test_single_layer import TestSingleLayer

import tensorflow as tf

from network.resnet_blocks import resnet_v1_block


class TestResnetV1Block(TestSingleLayer):
    def _generate_tf_model(self, sample):
        _, input_imsize, cin, cout, mid_channels, stride, kernel_size = sample

        inputs, nets = self._pad_before_input(
            [[1, input_imsize, input_imsize, cin]])

        net = nets[0]
        net = resnet_v1_block(
            net, [mid_channels, mid_channels, cout], stride, kernel_size)

        outputs = self._pad_after_output([net])
        return inputs, outputs
