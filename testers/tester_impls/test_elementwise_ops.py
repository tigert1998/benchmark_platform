import tensorflow as tf

from testers.tester import Tester, InferenceResult
from .test_single_layer import TestSingleLayer
from network.building_ops import channel_shuffle, global_pooling, depthwise_conv

from typing import Optional, Tuple


class TestAdd(TestSingleLayer):
    def _generate_tf_model(self, sample):
        _, input_imsize, cin = sample

        inputs, nets = self._pad_before_input(
            [[1, input_imsize, input_imsize, cin]] * 2)

        net = nets[0] + nets[1]

        outputs = self._pad_after_output([net])
        return inputs, outputs


class TestConcat(TestSingleLayer):
    def _generate_tf_model(self, sample):
        _, input_imsize, first_cin, second_cin = sample

        inputs, nets = self._pad_before_input([
            [1, input_imsize, input_imsize, first_cin],
            [1, input_imsize, input_imsize, second_cin]
        ])

        net = tf.concat(nets, axis=-1)

        outputs = self._pad_after_output([net])
        return inputs, outputs


class TestGlobalPooling(TestSingleLayer):
    def _generate_tf_model(self, sample):
        _, input_imsize, cin = sample

        inputs, nets = self._pad_before_input(
            [[1, input_imsize, input_imsize, cin]])

        net = nets[0]
        net = global_pooling(net)

        outputs = self._pad_after_output([net])
        return inputs, outputs


class TestShuffle(TestSingleLayer):
    def _generate_tf_model(self, sample):
        _, input_imsize, cin, num_groups = sample

        inputs, nets = self._pad_before_input(
            [[1, input_imsize, input_imsize, cin]])
        net = nets[0]

        net = channel_shuffle(net, num_groups)

        outputs = self._pad_after_output([net])
        return inputs, outputs


class TestActivation(Tester):
    @staticmethod
    def default_settings():
        return {
            **Tester.default_settings(),
            "min": 2,
            "max": 10
        }

    def add_layer(
        self, net: tf.Tensor,
        op: Optional[str], dwconv: bool
    ) -> tf.Tensor:
        assert op is not None or dwconv
        assert op in [None, "relu", "relu6", "swish", "hardswish", "sigmoid"]

        if dwconv:
            net = depthwise_conv(net, 1, 3)

        if op == "relu":
            net = tf.nn.relu(net)
        elif op == "relu6":
            net = tf.nn.relu6(net)
        elif op == "swish":
            net = tf.nn.swish(net)
        elif op == "hardswish":
            net = tf.nn.relu6(tf.math.add(net, 3)) * (1. / 6.) * net
        elif op == "sigmoid":
            net = tf.math.sigmoid(net)

        return net

    def _fetch_overall_latency(
        self,
        op: Optional[str], input_imsize: int, cin: int,
        dwconv: bool, n: int
    ) -> (float, float):
        tf.reset_default_graph()
        input_tensor = tf.placeholder(
            name="input_im_0",
            dtype=tf.float32,
            shape=(1, input_imsize, input_imsize, cin)
        )
        net = input_tensor

        for _ in range(n):
            net = self.add_layer(net, op, dwconv)

        output_tensor = net

        bench = "+".join((["dwconv"] if dwconv else []) +
                         ([op] if op is not None else []))
        model_path = "model_{}x_{}".format(n, bench)

        input_size_list = self.inference_sdk.generate_model(
            model_path,
            [input_tensor], [output_tensor]
        )
        res = self.inference_sdk.fetch_results(
            self.connection, model_path, input_size_list, self.benchmark_model_flags
        )
        return (res.avg_ms, res.std_ms)

    def _process_inference_result(self, result: InferenceResult):
        a, b = self.settings["min"], self.settings["max"]
        dic = result.profiling_details
        ret = {}

        ret["activation"] = \
            (dic[b]["activation"][0] -
             dic[a]["activation"][0]) / (b - a)
        ret["fused_activation"] = \
            ((dic[b]["dwconv+activation"][0] -
              dic[a]["dwconv+activation"][0]) -
             (dic[b]["dwconv"][0] -
              dic[a]["dwconv"][0])
             ) / (b - a)

        for n in [a, b]:
            for bench in ["activation", "dwconv", "dwconv+activation"]:
                name = "{}x_{}".format(n, bench)
                for i, metric in enumerate(["avg", "std"]):
                    ret["{}_{}_ms".format(name, metric)] = dic[n][bench][i]

        return ret

    def _test_sample(self, sample):
        op, input_imsize, cin = sample

        result_dic = dict()

        for n in [self.settings["min"], self.settings["max"]]:
            result_dic[n] = {}
            result_dic[n]["activation"] = self._fetch_overall_latency(
                op, input_imsize, cin, False, n
            )
            result_dic[n]["dwconv"] = self._fetch_overall_latency(
                None, input_imsize, cin, True, n
            )
            result_dic[n]["dwconv+activation"] = self._fetch_overall_latency(
                op, input_imsize, cin, True, n
            )

        return InferenceResult(
            avg_ms=None, std_ms=None,
            profiling_details=result_dic,
            layerwise_info=None
        )
