from testers.tester import Tester

import tensorflow as tf


class TestDwconv(Tester):
    @staticmethod
    def _get_metrics_titles():
        return ["latency_ms", "std_ms"]

    def _generate_model(self, sample):
        model_path = "model"
        _, _, input_imsize, cin, _, _, _, stride, kernel_size = sample
        tf.reset_default_graph()
        input_im = tf.placeholder(
            name="input_im", dtype=tf.float32,
            shape=(1, input_imsize, input_imsize, cin))

        net = tf.nn.depthwise_conv2d(
            input_im,
            filter=tf.get_variable(
                "dwconv_filter", [kernel_size, kernel_size, cin, 1], dtype=tf.float32, trainable=True),
            strides=[1, stride, stride, 1],
            padding='SAME',
            rate=[1, 1],
            name='the_dwconv'
        )

        self.inference_sdk.generate_model(model_path, [input_im], [net])
        return model_path

    def _test_sample(self, sample):
        model_path = self._generate_model(sample)
        results = self.inference_sdk.fetch_results(
            self.adb_device_id, model_path, self.benchmark_model_flags)
        return [results.avg_ms, results.std_ms]
