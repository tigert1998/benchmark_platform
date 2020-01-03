from testers.tester import Tester

import tensorflow as tf
from .utils import append_layerwise_info


class TestFc(Tester):
    def _test_sample(self, sample):
        _, _,  cin, cout, _, _ = sample
        tf.reset_default_graph()
        input_im = tf.placeholder(
            name="input_im", dtype=tf.float32, shape=(1, cin))
        net = tf.keras.layers.Dense(units=cout, name="the_fc")(input_im)
        self.inference_sdk.generate_model("model", [input_im], [net])

        input_size_list = input_im.get_shape().as_list()
        results = self.inference_sdk.fetch_results(
            self.adb_device_id, "model", input_size_list, self.benchmark_model_flags)
        return append_layerwise_info({
            "latency_ms": results.avg_ms,
            "std_ms": results.std_ms
        }, results.layerwise_info)
