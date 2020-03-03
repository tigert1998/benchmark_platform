from .test_single_layer import TestSingleLayer

import tensorflow as tf

from network.shufflenet_units import shufflenet_v1_unit, shufflenet_v2_unit


class TestShufflenetV1Unit(TestSingleLayer):
    def _generate_tf_model(self, sample):
        _, input_imsize, cin, cout, num_groups, mid_channels, stride, kernel_size = sample

        inputs, nets = self._pad_before_input(
            [[1, input_imsize, input_imsize, cin]])

        net = nets[0]
        net = shufflenet_v1_unit(
            net, num_groups, mid_channels, stride, kernel_size, cout)

        outputs = self._pad_after_output([net])
        return inputs, outputs


class TestShufflenetV2Unit(TestSingleLayer):
    def _generate_tf_model(self, sample):
        _, input_imsize, cin, stride, kernel_size = sample

        inputs, nets = self._pad_before_input(
            [[1, input_imsize, input_imsize, cin]])

        net = nets[0]
        net = shufflenet_v2_unit(net, stride, kernel_size)

        outputs = self._pad_after_output([net])
        return inputs, outputs
