from .test_single_layer import TestSingleLayer

import tensorflow as tf

from network.mbnet_blocks import mbnet_v1_block, mbnet_v2_block


class TestMbnetV1Block(TestSingleLayer):
    def _generate_tf_model(self, sample):
        _, input_imsize, cin, cout, stride, kernel_size = sample

        inputs, nets = self._pad_before_input(
            [[1, input_imsize, input_imsize, cin]])
        net = nets[0]

        net = mbnet_v1_block(net, stride, kernel_size, cout)

        outputs = self._pad_after_output([net])
        return inputs, outputs


class TestMbnetV2Block(TestSingleLayer):
    def _generate_tf_model(self, sample):
        _, input_imsize, cin, cout, with_se, stride, kernel_size = sample
        if with_se:
            se_mid_channels = cin // 4
        else:
            se_mid_channels = None

        inputs, nets = self._pad_before_input(
            [[1, input_imsize, input_imsize, cin]])
        net = nets[0]

        net = mbnet_v2_block(
            net, 6, stride, kernel_size,
            cout, se_mid_channels=se_mid_channels
        )

        outputs = self._pad_after_output([net])
        return inputs, outputs
