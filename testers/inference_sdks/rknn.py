import os

import tensorflow as tf
import numpy as np
import csv
import json
import shutil
from typing import List


from rknn.api import RKNN

from .inference_sdk import InferenceSdk, InferenceResult
from .utils import rfind_assign_float, rfind_assign_int
from utils.utils import concatenate_flags, rm_ext
from utils.stat import Stat
from utils.connection import Connection


class Rknn(InferenceSdk):
    @staticmethod
    def default_settings():
        return {
            **InferenceSdk.default_settings(),
            "rknn_target": "rk1808",
            "quantization": ""
        }

    @staticmethod
    def default_flags():
        return {
            **InferenceSdk.default_flags(),
            "num_runs": 50,
            "enable_op_profiling": True,
        }

    def __init__(self, settings={}):
        super().__init__(settings)
        self.quantization = self.settings["quantization"]
        assert self.quantization in [
            "", "asymmetric_quantized-u8",
            "dynamic_fixed_point-8",
            "dynamic_fixed_point-16"
        ]

    @staticmethod
    def _build_rknn_fake_quantization_dataset(input_size_list: List[List[int]]):
        # batch dimension is removed

        dataset_path = "fake_dataset"
        os.makedirs(dataset_path, exist_ok=True)
        idx_file = "{}/index.txt".format(dataset_path)

        with open(idx_file, "w") as f:
            for i in range(10):
                for j, input_size in enumerate(input_size_list):
                    npy_filename = os.path.abspath(
                        "{}/{}.{}.npy".format(dataset_path, i, j))
                    np.save(npy_filename, np.random.randn(1, *input_size))
                    f.write("{} ".format(npy_filename))
                f.write("\n")

        return idx_file

    def generate_model(self, path, inputs, outputs):
        outputs_ops_names = [o.op.name for o in outputs]

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            constant_graph = tf.graph_util.convert_variables_to_constants(
                sess, sess.graph_def, outputs_ops_names)
            with tf.gfile.FastGFile(path + '.pb', mode='wb') as f:
                f.write(constant_graph.SerializeToString())

            rknn_convert_config = {
                "inputs": [i.op.name for i in inputs],
                "input_size_list": [i.get_shape().as_list()[1:] for i in inputs],
                "outputs": outputs_ops_names
            }

        # remember to modify RKNN.__init__
        rknn = RKNN()
        rknn.config(batch_size=1, quantized_dtype=self.quantization)
        assert 0 == rknn.load_tensorflow(
            path + '.pb',
            inputs=rknn_convert_config["inputs"],
            input_size_list=rknn_convert_config["input_size_list"],
            outputs=rknn_convert_config["outputs"]
        )

        if self.quantization == "":
            assert 0 == rknn.build(
                do_quantization=False,
                dataset="",
                pre_compile=False
            )
        else:
            dataset = self._build_rknn_fake_quantization_dataset(
                rknn_convert_config["input_size_list"])
            assert 0 == rknn.build(
                do_quantization=True,
                dataset=dataset,
                pre_compile=False
            )

        assert 0 == rknn.export_rknn(path + '.rknn')
        rknn.release()

    def _fetch_results_on_soc(self,
                              connection: Connection, model_path,
                              input_size_list: List[List[int]], flags) -> InferenceResult:
        model_basename = os.path.basename(model_path)

        model_folder = "/mnt/sdcard/channel_benchmark"
        benchmark_model_folder = "/data/local/tmp/rknn_benchmark_model"
        connection.push(model_path + ".rknn", model_folder)

        cmd = "LD_LIBRARY_PATH={}/lib64 {}/rknn_benchmark_model {}".format(
            benchmark_model_folder,
            benchmark_model_folder, concatenate_flags({
                "benchmark_type": "latency",
                "model_path": "{}/{}.rknn".format(model_folder, model_basename),
                "op_profiling_dump_path": "{}/op_profiling.csv".format(model_folder),
                **flags
            }))
        print(cmd)

        result_str = connection.shell(cmd)
        print(result_str)
        assert result_str.find("TIMEOUT") == -1

        if rfind_assign_int(result_str, "count") >= 2:
            std_ms = rfind_assign_float(result_str, 'std')
            avg_ms = rfind_assign_float(result_str, 'avg')
        else:
            std_ms = 0
            avg_ms = rfind_assign_float(result_str, 'curr')

        if flags["enable_op_profiling"]:
            connection.pull("{}/op_profiling.csv".format(model_folder), ".")
            layerwise_info = []
            with open("op_profiling.csv", "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    layerwise_info.append({
                        "name": "{}_{}_{}_{}_{}_{}".format(
                            row["Layer id"], row["Name"], row["Operation id"],
                            row["Operator"], row["Target"], row["Uid"]
                        ),
                        "time": {
                            "avg_ms": float(row["avg(us)"]) / 1e3,
                            "std_ms": float(row["std"]) / 1e3
                        }
                    })
        else:
            layerwise_info = None

        return InferenceResult(avg_ms=avg_ms, std_ms=std_ms, profiling_details=None, layerwise_info=layerwise_info)

    def _fetch_results_with_py_api(self,
                                   connection: Connection, model_path,
                                   input_size_list: List[List[int]], flags) -> InferenceResult:
        rknn = RKNN()

        assert self.settings["enable_op_profiling"]
        assert 0 == rknn.load_rknn(model_path + ".rknn")
        assert 0 == rknn.init_runtime(
            target=self.settings["rknn_target"], perf_debug=True)

        inputs = [np.random.rand(*size).astype(np.float32)
                  for size in input_size_list]

        stat = Stat()
        layerwise_info = None
        # FIXME
        assert False
        for i in range(flags["num_runs"]):
            result = rknn.eval_perf(inputs=inputs, is_print=False)
            stat.update(result["total_time"] / 1e3)
            result_layer = result["layers"]
            if layerwise_info is None:
                layerwise_info = []
                for value in result_layer.values():
                    layerwise_info.append({
                        "name": "{}_{}".format(value["name"], value["uid"]),
                        "time": Stat()
                    })
                    layerwise_info[-1]["time"].update(value["time"] / 1e3)
            else:
                for i, (key, value) in enumerate(result_layer.items()):
                    assert layerwise_info[i]["name"] == \
                        "{}_{}".format(value["name"], value["uid"])
                    layerwise_info[i]["time"].update(value["time"] / 1e3)

        for dic in layerwise_info:
            layer_stat = dic.pop("time")
            dic["time"] = {
                "avg_ms": layer_stat.avg(),
                "std_ms": layer_stat.std()
            }

        rknn.release()

        return InferenceResult(avg_ms=stat.avg(), std_ms=stat.std(), profiling_details=None, layerwise_info=layerwise_info)

    def _fetch_results(self,
                       connection: Connection, model_path,
                       input_size_list: List[List[int]], flags) -> InferenceResult:
        if self.settings["rknn_target"] is None:
            return self._fetch_results_on_soc(connection, model_path, input_size_list, flags)
        else:
            return self._fetch_results_with_py_api(connection, model_path, input_size_list, flags)
