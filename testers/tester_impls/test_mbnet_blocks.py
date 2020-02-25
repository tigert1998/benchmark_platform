from .test_single_layer import TestSingleLayer

import tensorflow as tf

from network.mbnet_blocks import mbnet_v1_block, mbnet_v2_block


class TestMbnetV1Block(TestSingleLayer):
    def _generate_model(self, sample):
        model_path = "model"
        _, input_imsize, cin, cout, stride, kernel_size = sample

        tf.reset_default_graph()
        input_im = tf.placeholder(
            name="input_im", dtype=tf.float32,
            shape=(1, input_imsize, input_imsize, cin))

        net = mbnet_v1_block(input_im, stride, kernel_size, cout)

        self.inference_sdk.generate_model(model_path, [input_im], [net])
        return model_path, [input_im.get_shape().as_list()]


class TestMbnetV2Block(TestSingleLayer):
    def _generate_model(self, sample):
        model_path = "model"
        _, input_imsize, cin, cout, with_se, stride, kernel_size = sample

        tf.reset_default_graph()

        input_im = tf.placeholder(
            name="input_im", dtype=tf.float32,
            shape=(1, input_imsize, input_imsize, cin))

        if with_se:
            se_mid_channels = cin // 4
        else:
            se_mid_channels = None
        net = mbnet_v2_block(input_im, 6, stride, kernel_size,
                             cout, se_mid_channels=se_mid_channels)

        self.inference_sdk.generate_model(model_path, [input_im], [net])
        return model_path, [input_im.get_shape().as_list()]
