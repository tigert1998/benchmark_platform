from testers.tester import Tester

import tensorflow as tf


class TestDilatedConv(Tester):
    def _generate_model(self, sample):
        model_path = "model"

        _, input_imsize, cin, cout, dilation, stride, kernel_size = sample

        tf.reset_default_graph()
        input_im = tf.placeholder(
            name="input_im", dtype=tf.float32, shape=(1, input_imsize, input_imsize, cin))
        net = tf.keras.layers.Conv2D(
            filters=cout,
            kernel_size=[kernel_size, kernel_size],
            strides=[stride, stride],
            padding='same',
            dilation_rate=dilation,
            name="the_dilated_conv")(input_im)
        self.inference_sdk.generate_model(model_path, [input_im], [net])
        return model_path, input_im.get_shape().as_list()

    def _test_sample(self, sample):
        model_path, input_size_list = self._generate_model(sample)
        return self.inference_sdk.fetch_results(
            self.connection, model_path, input_size_list, self.benchmark_model_flags)
