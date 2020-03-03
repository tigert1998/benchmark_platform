from .test_single_layer import TestSingleLayer

import tensorflow as tf


class TestFc(TestSingleLayer):
    def _generate_tf_model(self, sample):
        _, _, cin, cout, _, _ = sample

        inputs, nets = self._pad_before_input([[1, cin]])

        net = nets[0]
        net = tf.keras.layers.Dense(units=cout, name="the_fc")(net)

        outputs = self._pad_after_output([net])
        return inputs, outputs
