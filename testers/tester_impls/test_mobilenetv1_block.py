from testers.tester import Tester

import tensorflow as tf


class TestMobilenetv1Block(Tester):
    def _generate_model(self, sample):
        model_path = "model"
        _, input_imsize, cin, cout, stride, kernel_size = sample

        tf.reset_default_graph()
        input_im = tf.placeholder(
            name="input_im", dtype=tf.float32,
            shape=(1, input_imsize, input_imsize, cin))

        net = tf.nn.relu6(tf.nn.depthwise_conv2d(
            input_im,
            filter=tf.get_variable(
                "dwconv_filter", [kernel_size, kernel_size, cin, 1],
                dtype=tf.float32, trainable=True),
            strides=[1, stride, stride, 1],
            padding='SAME',
            rate=[1, 1],
            name='the_dwconv'
        ))
        net = tf.nn.relu6(tf.keras.layers.Conv2D(
            filters=cout,
            kernel_size=[1, 1],
            strides=[1, 1],
            padding='same', name="the_conv")(net))

        self.inference_sdk.generate_model(model_path, [input_im], [net])
        return model_path, input_im.get_shape().as_list()

    def _test_sample(self, sample):
        model_path, input_size_list = self._generate_model(sample)
        return self.inference_sdk.fetch_results(
            self.connection, model_path, input_size_list, self.benchmark_model_flags)
