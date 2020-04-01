from .test_single_layer import TestSingleLayer

import tensorflow as tf

from network.building_ops import mix_conv


class TestMixConv(TestSingleLayer):
    def _generate_tf_model(self, sample):
        _, input_imsize, cin, cout, num_groups, stride = sample
        assert cin == cout

        inputs, nets = self._pad_before_input(
            [[1, input_imsize, input_imsize, cin]])
        net = nets[0]

        net = mix_conv(net, num_groups, stride)

        outputs = self._pad_after_output([net])
        return inputs, outputs
