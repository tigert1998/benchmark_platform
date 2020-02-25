from .test_single_layer import TestSingleLayer

import tensorflow as tf

from network.building_ops import grouped_conv


class TestGconv(TestSingleLayer):
    def _generate_model(self, sample):
        model_path = "model"

        _, input_imsize, cin, cout, num_groups, stride, kernel_size = sample

        tf.reset_default_graph()
        input_im = tf.placeholder(
            name="input_im", dtype=tf.float32, shape=(1, input_imsize, input_imsize, cin))

        net = grouped_conv(input_im, num_groups, stride, kernel_size, cout)

        self.inference_sdk.generate_model(model_path, [input_im], [net])
        return model_path, [input_im.get_shape().as_list()]
