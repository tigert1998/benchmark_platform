from .test_single_layer import TestSingleLayer

import tensorflow as tf

from network.dense_blocks import dense_block


class TestDenseBlock(TestSingleLayer):
    def _generate_model(self, sample):
        model_path = "model"
        _, input_imsize, cin, growth_rate, num_layers, kernel_size = sample

        tf.reset_default_graph()

        input_im = tf.placeholder(
            name="input_im", dtype=tf.float32,
            shape=(1, input_imsize, input_imsize, cin))

        net = dense_block(
            input_im, num_layers, True, kernel_size, growth_rate)

        self.inference_sdk.generate_model(model_path, [input_im], [net])
        return model_path, [input_im.get_shape().as_list()]
