from .test_single_layer import TestSingleLayer

import tensorflow as tf


class TestConv(TestSingleLayer):
    def _generate_model(self, sample):
        model_path = "model"
        _, _, input_imsize, cin, cout, _, _, stride, kernel_size = sample
        tf.reset_default_graph()
        input_im = tf.placeholder(
            name="input_im", dtype=tf.float32, shape=(1, input_imsize, input_imsize, cin))
        net = tf.keras.layers.Conv2D(filters=cout,
                                     kernel_size=[kernel_size, kernel_size],
                                     strides=[stride, stride],
                                     padding='same', name="the_conv")(input_im)
        self.inference_sdk.generate_model(model_path, [input_im], [net])
        return model_path, [input_im.get_shape().as_list()]
