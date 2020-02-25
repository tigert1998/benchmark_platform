import os

import tensorflow as tf
import numpy as np
import csv
import json
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
            "pre_compile": False
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
        if self.settings["pre_compile"]:
            import docker
            self.container: docker.models.containers.Container =\
                docker.from_env().containers.list()[0]

    def generate_model(self, path, inputs, outputs):
        outputs_ops_names = [o.op.name for o in outputs]

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            constant_graph = tf.graph_util.convert_variables_to_constants(
                sess, sess.graph_def, outputs_ops_names)
            with tf.gfile.FastGFile(path + '.pb', mode='wb') as f:
                f.write(constant_graph.SerializeToString())

            docker_path = os.path.abspath(path).split(os.path.sep)
            docker_path = docker_path[docker_path.index("benchmark_platform"):]
            docker_path = "/" + "/".join(["playground"] + docker_path)
            rknn_convert_config = {
                "path": docker_path,
                "inputs": [i.op.name for i in inputs],
                "input_size_list": [i.get_shape().as_list()[1:] for i in inputs],
                "outputs": outputs_ops_names
            }

        if not self.settings["pre_compile"]:
            # remember to modify RKNN.__init__
            rknn = RKNN()
            rknn.config(batch_size=1)
            assert 0 == rknn.load_tensorflow(
                path + '.pb',
                inputs=rknn_convert_config["inputs"],
                input_size_list=rknn_convert_config["input_size_list"],
                outputs=rknn_convert_config["outputs"]
            )
            assert 0 == rknn.build(do_quantization=False,
                                   dataset="", pre_compile=False)
            assert 0 == rknn.export_rknn(path + '.rknn')
            rknn.release()
        else:
            host_config_path = os.path.abspath(__file__).split(os.path.sep)
            host_config_path = os.path.sep.join(host_config_path[:host_config_path.index(
                "benchmark_platform")])
            host_config_path = os.path.join(
                host_config_path, "rknn_convert_tools/config.json")
            with open(host_config_path, "w") as f:
                f.write(json.dumps(rknn_convert_config))
            convert_cmd = "{} run -n rknn python /playground/rknn_convert_tools/main.py".format(
                "/root/anaconda3/bin/conda")
            result = self.container.exec_run(convert_cmd)
            assert 0 == result.exit_code
            print(result.output.decode('utf-8'))

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

        connection.pull("{}/op_profiling.csv".format(model_folder), ".")
        layerwise_info = []
        with open("op_profiling.csv") as f:
            reader = csv.DictReader(f)
            for row in reader:
                layerwise_info.append({
                    "name": "{}_{}".format(row["Name"], row["Uid"]),
                    "time": {
                        "avg_ms": float(row["avg(us)"]) / 1e3,
                        "std_ms": float(row["std"]) / 1e3
                    }
                })

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
