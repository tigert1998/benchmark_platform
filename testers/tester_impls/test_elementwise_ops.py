from .test_single_layer import TestSingleLayer

import tensorflow as tf

from network.building_ops import channel_shuffle, global_pooling


class TestAdd(TestSingleLayer):
    def _generate_tf_model(self, sample):
        _, input_imsize, cin = sample

        inputs, nets = self._pad_before_input(
            [[1, input_imsize, input_imsize, cin]] * 2)

        net = nets[0] + nets[1]

        outputs = self._pad_after_output([net])
        return inputs, outputs


class TestConcat(TestSingleLayer):
    def _generate_tf_model(self, sample):
        _, input_imsize, first_cin, second_cin = sample

        inputs, nets = self._pad_before_input([
            [1, input_imsize, input_imsize, first_cin],
            [1, input_imsize, input_imsize, second_cin]
        ])

        net = tf.concat(nets, axis=-1)

        outputs = self._pad_after_output([net])
        return inputs, outputs


class TestGlobalPooling(TestSingleLayer):
    def _generate_tf_model(self, sample):
        _, input_imsize, cin = sample

        inputs, nets = self._pad_before_input(
            [[1, input_imsize, input_imsize, cin]])

        net = nets[0]
        net = global_pooling(net)

        outputs = self._pad_after_output([net])
        return inputs, outputs


class TestShuffle(TestSingleLayer):
    def _generate_tf_model(self, sample):
        _, input_imsize, cin, num_groups = sample

        inputs, nets = self._pad_before_input(
            [[1, input_imsize, input_imsize, cin]])
        net = nets[0]

        net = channel_shuffle(net, num_groups)

        outputs = self._pad_after_output([net])
        return inputs, outputs


class TestActivation(TestSingleLayer):
    def _generate_tf_model(self, sample):
        op, input_imsize, cin = sample
        assert op in ["relu", "relu6", "swish", "sigmoid"]

        inputs, nets = self._pad_before_input(
            [[1, input_imsize, input_imsize, cin]])
        net = nets[0]

        if op == "relu":
            net = tf.nn.relu(net)
        elif op == "relu6":
            net = tf.nn.relu6(net)
        elif op == "swish":
            net = tf.math.multiply(net, tf.math.sigmoid(net))
        elif op == "sigmoid":
            net = tf.math.sigmoid(net)

        outputs = self._pad_after_output([net])
        return inputs, outputs
