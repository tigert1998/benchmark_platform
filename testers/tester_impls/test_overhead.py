from testers.tester import Tester

import tensorflow as tf
import numpy as np

from typing import Tuple, List

from testers.inference_sdks.inference_sdk import InferenceResult


class TestOverhead(Tester):
    @staticmethod
    def _conv_add_layer(net: tf.Tensor) -> tf.Tensor:
        cin = net.get_shape().as_list()[-1]
        return tf.keras.layers.Conv2D(
            filters=cin,
            kernel_size=[1, 1],
            strides=[1, 1],
            padding='same'
        )(net)

    @staticmethod
    def _dwconv_add_layer(net: tf.Tensor) -> tf.Tensor:
        cin = net.get_shape().as_list()[-1]
        return tf.nn.depthwise_conv2d(
            net,
            filter=tf.constant(
                np.random.randn(3, 3, cin, 1).astype(np.float32)
            ),
            strides=[1, 1, 1, 1],
            padding='SAME',
            rate=[1, 1],
        )

    @staticmethod
    def _id_add_layer(net: tf.Tensor) -> tf.Tensor:
        return tf.identity(net)

    @staticmethod
    def default_settings():
        return {
            **Tester.default_settings(),
            "op": "conv",
            "min": 10,
            "max": 100
        }

    def __init__(self, settings={}):
        super().__init__(settings)
        dic = {
            "conv": self._conv_add_layer,
            "dwconv": self._dwconv_add_layer,
            "id": self._id_add_layer
        }
        assert self.settings["op"] in dic
        self.add_layer = dic[self.settings["op"]]
        self.is_id = self.settings["op"] == "id"

    def fetch_overall_latency(self, imsize: int, cin: int, n: int) -> (float, float):
        """Fetch overall latency of a stacked model
        Returns:
            (avg_ms, std_ms)
        """

        tf.reset_default_graph()
        input_im = tf.placeholder(
            name="input_im",
            dtype=tf.float32,
            shape=[1, imsize, imsize, cin]
        )
        net = input_im
        for _ in range(n):
            net = self.add_layer(net)

        model_path = "model_{}x_{}".format(n, self.settings["op"])
        input_size_list = self.inference_sdk.generate_model(
            model_path,
            [input_im], [net]
        )
        res = self.inference_sdk.fetch_results(
            self.connection, model_path, input_size_list, self.benchmark_model_flags
        )
        return (res.avg_ms, res.std_ms)

    def _process_inference_result(self, result: InferenceResult):
        ret = {}
        for i, (avg_ms, std_ms) in result.profiling_details["results"].items():
            ret["{}_avg_ms".format(i)] = avg_ms
            ret["{}_std_ms".format(i)] = std_ms
        ret["estimated"] = result.profiling_details["estimated"]
        return ret

    def _test_sample(self, sample):
        imsize, cin = sample

        if self.is_id:
            todos = [1]
        else:
            todos = [self.settings["min"], self.settings["max"]]

        results = {
            i: self.fetch_overall_latency(imsize, cin, i)
            for i in todos
        }

        if self.is_id:
            estimated_overhead = results[1][0] / 2
        else:
            max_res = results[self.settings["max"]]
            min_res = results[self.settings["min"]]
            layer_latency = (max_res[0] - min_res[0]) / \
                (self.settings["max"] - self.settings["min"])
            estimated_overhead = (
                max_res[0] - layer_latency * self.settings["max"]) / 2

        return InferenceResult(
            avg_ms=None, std_ms=None,
            profiling_details={
                "results": results,
                "estimated": estimated_overhead
            },
            layerwise_info=None
        )
