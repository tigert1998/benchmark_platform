from .inference_sdk import InferenceSdk, InferenceResult
from utils.connection import Connection
from utils.utils import concatenate_flags, rm_ext
from .utils import rfind_assign_float

import numpy as np
import os
from typing import List
import tensorflow as tf
import time

from .tflite import Tflite


class Tpu(InferenceSdk):
    @staticmethod
    def default_settings():
        return {
            **InferenceSdk.default_settings(),
            "edgetpu_compiler_path": "edgetpu_compiler",
            "libedgetpu_path": "libedgetpu.so.1"
        }

    @staticmethod
    def default_flags():
        return {
            **InferenceSdk.default_flags(),
            "warmup_runs": 1,
            "num_runs": 30
        }

    def __init__(self, settings={}):
        super().__init__(settings)
        import tflite_runtime.interpreter as tflite

        self.tflite_model_generator = Tflite({"quantization": "int"})
        self.edgetpu_compiler_path = self.settings["edgetpu_compiler_path"]
        self.delegate = tflite.load_delegate(self.settings["libedgetpu_path"])

    def generate_model(self, path, inputs, outputs):
        self.tflite_model_generator.generate_model(path, inputs, outputs)
        cmds = [
            "{} {}.tflite -o {}".format(
                self.edgetpu_compiler_path,
                path, os.path.dirname(path)
            ),
            "mv {}.tflite {}_int_quant.tflite".format(path, path),
            "mv {}_edgetpu.tflite {}.tflite".format(path, path)
        ]
        for cmd in cmds:
            self.tflite_model_generator.local_connection.shell(cmd)

    def _fetch_results(self,
                       connection: Connection, model_path: str,
                       input_size_list: List[List[int]], benchmark_model_flags) -> InferenceResult:
        import tflite_runtime.interpreter as tflite

        interpreter = tflite.Interpreter(
            model_path=model_path + ".tflite",
            experimental_delegates=[self.delegate])

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        interpreter.allocate_tensors()

        for input_detail in input_details:
            interpreter.set_tensor(
                input_detail["index"],
                np.random.randn(
                    *input_detail["shape"]).astype(input_detail["dtype"])
            )

        for _ in range(benchmark_model_flags["warmup_runs"]):
            interpreter.invoke()

        num_runs = benchmark_model_flags["num_runs"]

        sum_ms = 0
        square_sum_ms = 0
        for _ in range(num_runs):
            start = time.time()
            interpreter.invoke()
            duration = 1000 * (time.time() - start)
            sum_ms += duration
            square_sum_ms += duration ** 2

        avg_ms = sum_ms / num_runs
        std_ms = square_sum_ms / num_runs - avg_ms ** 2

        return InferenceResult(avg_ms=avg_ms, std_ms=std_ms, profiling_details=None, layerwise_info=None)
