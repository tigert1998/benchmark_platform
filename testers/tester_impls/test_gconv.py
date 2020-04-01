from .test_single_layer import TestSingleLayer

import tensorflow as tf

from network.building_ops import grouped_conv


class TestGconv(TestSingleLayer):
    def _generate_tf_model(self, sample):
        _, input_imsize, cin, cout, num_groups, stride, kernel_size = sample

        inputs, nets = self._pad_before_input(
            [[1, input_imsize, input_imsize, cin]])
        net = nets[0]

        net = grouped_conv(net, num_groups, stride, kernel_size, cout)

        outputs = self._pad_after_output([net])
        return inputs, outputs
