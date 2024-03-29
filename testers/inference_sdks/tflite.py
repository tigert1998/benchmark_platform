import tensorflow as tf
import numpy as np

import os
import subprocess
import shutil
from typing import List

from .inference_sdk import InferenceSdk, InferenceResult
from .utils import rfind_assign_float, table_try_float, rfind_assign_int
from utils.utils import concatenate_flags, search_python_script
from utils.connection import Connection
from utils.tf_model_utils import to_saved_model


class Tflite(InferenceSdk):
    @staticmethod
    def default_settings():
        return {
            **InferenceSdk.default_settings(),
            "benchmark_model_path": None,
            "taskset": "f0",
            "quantization": ""
        }

    def __init__(self, settings={}):
        super().__init__(settings)
        self.local_connection = Connection()

    @staticmethod
    def default_flags():
        return {
            "use_gpu": None
        }

    def generate_model(self, path, inputs, outputs):
        path = os.path.abspath(path)
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            to_saved_model(
                sess, inputs, outputs, path,
                replace_original_dir=True
            )

        output_path = path + ".tflite"
        if os.path.isfile(output_path):
            os.remove(output_path)
        cmd = "conda activate tf2; python {} {}".format(
            search_python_script("conversion/to_tflite", __file__),
            concatenate_flags({
                "saved_model_path": path,
                "quantization": self.settings["quantization"],
                "output_path": output_path
            })
        )
        print(cmd)
        self.local_connection.shell(cmd)

    def _launch_benchmark(self, connection: Connection, model_path: str, flags):
        model_basename = os.path.basename(model_path)

        model_folder = "/mnt/sdcard/channel_benchmark"
        connection.push(model_path + ".tflite", model_folder)

        if self.settings["taskset"] is None:
            taskset_prefix = ""
        else:
            taskset_prefix = "taskset " + self.settings["taskset"].strip()

        cmd = "{} {} {}".format(
            taskset_prefix,
            self.settings["benchmark_model_path"],
            concatenate_flags(
                {
                    "graph": "{}/{}.tflite".format(model_folder, model_basename),
                    **flags
                })
        )
        print(cmd.strip())
        return connection.shell(cmd)

    def _fetch_results(self,
                       connection: Connection, model_path: str,
                       input_size_list: List[List[int]], flags) -> InferenceResult:
        result_str = self._launch_benchmark(connection, model_path, flags)

        if rfind_assign_int(result_str, 'count') >= 2:
            std_ms = rfind_assign_float(result_str, 'std') / 1e3
            avg_ms = rfind_assign_float(result_str, 'avg') / 1e3
        else:
            std_ms = 0
            avg_ms = rfind_assign_float(result_str, 'curr') / 1e3

        use_delegate = flags.get("use_nnapi", False) or flags.get(
            "use_gpu", False) or flags.get("use_legacy_nnapi", False)
        enable_op_profiling = flags.get("enable_op_profiling", False)

        if use_delegate or (not enable_op_profiling):
            return InferenceResult(avg_ms=avg_ms, std_ms=std_ms, profiling_details=None, layerwise_info=None)
        else:
            op_profiling = []
            started = False
            for line in result_str.split('\n'):
                if "Top by Computation Time" in line:
                    break
                if "Run Order" in line:
                    started = True
                    continue
                if started:
                    cells = list(filter(lambda x: len(x) >= 1, map(
                        lambda x: x.strip(), line.split('\t'))))
                    if len(cells) == 0:
                        continue
                    op_profiling.append(cells)

            op_profiling = table_try_float(op_profiling)
            for i in range(len(op_profiling[0])):
                s = op_profiling[0][i]
                assert s.startswith("[") and s.endswith("]")
                op_profiling[0][i] = s[1: -1].lower()

            name_idx = op_profiling[0].index("name")
            type_idx = op_profiling[0].index("node type")
            avg_ms_idx = op_profiling[0].index("avg ms")

            layerwise_info = []
            name_set = set()
            for i in range(1, len(op_profiling)):
                name = "{}_{}".format(
                    op_profiling[i][name_idx],
                    op_profiling[i][type_idx]
                )
                assert name not in name_set
                name_set.add(name)
                avg_ms = float(op_profiling[i][avg_ms_idx])
                layerwise_info.append({
                    "name": name,
                    "time": {
                        "avg_ms": avg_ms, "std_ms": np.nan
                    }
                })

            return InferenceResult(
                avg_ms=avg_ms, std_ms=std_ms,
                profiling_details=None,
                layerwise_info=layerwise_info
            )
