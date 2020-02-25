from .test_single_layer import TestSingleLayer

import tensorflow as tf


class TestFc(TestSingleLayer):
    def _generate_model(self, sample):
        model_path = "model"

        _, _, cin, cout, _, _ = sample
        tf.reset_default_graph()
        input_im = tf.placeholder(
            name="input_im", dtype=tf.float32, shape=(1, cin))
        net = tf.keras.layers.Dense(units=cout, name="the_fc")(input_im)
        self.inference_sdk.generate_model(model_path, [input_im], [net])

        return model_path, [input_im.get_shape().as_list()]
