from testers.tester import Tester

import tensorflow as tf


class TestFc(Tester):
    def _test_sample(self, sample):
        _, _,  cin, cout, _, _ = sample
        tf.reset_default_graph()
        input_im = tf.placeholder(
            name="input_im", dtype=tf.float32, shape=(1, cin))
        net = tf.keras.layers.Dense(units=cout, name="the_fc")(input_im)
        self.inference_sdk.generate_model("model", [input_im], [net])

        input_size_list = input_im.get_shape().as_list()
        return self.inference_sdk.fetch_results(
            self.connection, "model", input_size_list, self.benchmark_model_flags)
