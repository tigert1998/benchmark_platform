from testers.tester import Tester

import tensorflow as tf

from network.mbnet_blocks import mbnet_v1_block


class TestMbnetV1Block(Tester):
    def _generate_model(self, sample):
        model_path = "model"
        _, input_imsize, cin, cout, stride, kernel_size = sample

        tf.reset_default_graph()
        input_im = tf.placeholder(
            name="input_im", dtype=tf.float32,
            shape=(1, input_imsize, input_imsize, cin))

        net = mbnet_v1_block(input_im, stride, kernel_size, cout)

        self.inference_sdk.generate_model(model_path, [input_im], [net])
        return model_path, input_im.get_shape().as_list()

    def _test_sample(self, sample):
        model_path, input_size_list = self._generate_model(sample)
        return self.inference_sdk.fetch_results(
            self.connection, model_path, input_size_list, self.benchmark_model_flags)
