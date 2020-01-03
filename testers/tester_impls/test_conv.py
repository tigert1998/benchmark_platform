from testers.tester import Tester

import tensorflow as tf
from .utils import append_layerwise_info


class TestConv(Tester):
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
        return model_path, input_im.get_shape().as_list()

    def _test_sample(self, sample):
        model_path, input_size_list = self._generate_model(sample)
        results = self.inference_sdk.fetch_results(
            self.adb_device_id, model_path, input_size_list, self.benchmark_model_flags)
        return append_layerwise_info({
            "latency_ms": results.avg_ms,
            "std_ms": results.std_ms
        }, results.layerwise_info)
