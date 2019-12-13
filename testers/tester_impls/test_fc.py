from testers.tester import Tester

import tensorflow as tf


class TestFc(Tester):
    @staticmethod
    def _get_metrics_titles():
        return ["latency_ms", "std_ms"]

    def _test_sample(self, sample):
        _, _,  cin, cout, _, _ = sample
        tf.reset_default_graph()
        input_im = tf.placeholder(
            name="input_im", dtype=tf.float32, shape=(1, cin))
        net = tf.keras.layers.Dense(units=cout, name="the_fc")(input_im)
        self.inference_sdk.generate_model("model", [input_im], [net])
        results = self.inference_sdk.fetch_results(
            self.adb_device_id, "model", self.benchmark_model_flags)
        return [results.avg_ms, results.std_ms]