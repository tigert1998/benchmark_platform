from testers.tester import Tester

import tensorflow as tf


class TestGconv(Tester):
    def _generate_model(self, sample):
        model_path = "model"

        _, input_imsize, cin, cout, num_groups, stride, kernel_size = sample
        assert cin % num_groups == 0 and cout % num_groups == 0

        tf.reset_default_graph()
        input_im = tf.placeholder(
            name="input_im", dtype=tf.float32, shape=(1, input_imsize, input_imsize, cin))

        groups = [
            tf.keras.layers.Conv2D(
                filters=cout // num_groups,
                kernel_size=[kernel_size, kernel_size],
                strides=[stride, stride],
                padding="same",
                name="the_conv_{}".format(i)
            )(x) for i, x in zip(range(num_groups), tf.split(input_im, num_groups, axis=-1))
        ]
        net = tf.concat(groups, axis=-1, name="the_gconv")

        self.inference_sdk.generate_model(model_path, [input_im], [net])
        return model_path, input_im.get_shape().as_list()

    def _test_sample(self, sample):
        model_path, input_size_list = self._generate_model(sample)
        return self.inference_sdk.fetch_results(
            self.connection, model_path, input_size_list, self.benchmark_model_flags)
