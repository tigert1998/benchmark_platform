from .test_single_layer import TestSingleLayer

import tensorflow as tf

from network.resnet_blocks import resnet_v1_block


class TestResnetV1Block(TestSingleLayer):
    def _generate_model(self, sample):
        model_path = "model"
        _, input_imsize, cin, cout, mid_channels, stride, kernel_size = sample

        tf.reset_default_graph()

        input_im = tf.placeholder(
            name="input_im", dtype=tf.float32,
            shape=(1, input_imsize, input_imsize, cin))

        net = resnet_v1_block(
            input_im, [mid_channels, mid_channels, cout], stride, kernel_size)

        self.inference_sdk.generate_model(model_path, [input_im], [net])
        return model_path, [input_im.get_shape().as_list()]
