from testers.tester import Tester

import tensorflow as tf


class TestMobilenetv2Block(Tester):
    def _generate_model(self, sample):
        model_path = "model"
        _, input_imsize, cin, cout, stride, kernel_size = sample

        assert stride in [1, 2]

        depth_multiplier = 6

        tf.reset_default_graph()

        input_im = tf.placeholder(
            name="input_im", dtype=tf.float32,
            shape=(1, input_imsize, input_imsize, cin))

        net = tf.nn.relu6(tf.keras.layers.Conv2D(
            filters=cin * depth_multiplier,
            kernel_size=[1, 1],
            strides=[1, 1],
            padding='same', name="the_conv_0")(input_im))

        net = tf.nn.relu6(tf.nn.depthwise_conv2d(
            net,
            filter=tf.get_variable(
                "dwconv_filter",
                [kernel_size, kernel_size, cin*depth_multiplier, 1],
                dtype=tf.float32, trainable=True
            ),
            strides=[1, stride, stride, 1],
            padding='SAME',
            rate=[1, 1],
            name='the_dwconv'
        ))

        net = tf.keras.layers.Conv2D(
            filters=cout,
            kernel_size=[1, 1],
            strides=[1, 1],
            padding='same', name="the_conv_1"
        )(net)

        if stride == 1:
            assert cin == cout
            net = input_im + net

        self.inference_sdk.generate_model(model_path, [input_im], [net])
        return model_path, input_im.get_shape().as_list()

    def _test_sample(self, sample):
        model_path, input_size_list = self._generate_model(sample)
        return self.inference_sdk.fetch_results(
            self.connection, model_path, input_size_list, self.benchmark_model_flags)
