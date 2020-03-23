from .test_single_layer import TestSingleLayer

import tensorflow as tf
import numpy as np


class TestMatmul(TestSingleLayer):
    def _generate_tf_model(self, sample):
        n, = sample

        inputs, nets = self._pad_before_input([[1, n, n]])

        net = nets[0]
        net = tf.linalg.matmul(net, tf.constant(np.random.randn(n, n)))

        outputs = self._pad_after_output([net])
        return inputs, outputs
