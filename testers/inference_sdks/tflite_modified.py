from .tflite import Tflite
from .inference_sdk import InferenceResult
from .utils import rfind_assign_float, rfind_assign_int, rfind_assign
from utils.connection import Connection

import os
from typing import List


class TfliteModified(Tflite):
    @staticmethod
    def default_flags():
        return {
            **Tflite.default_flags(),
            "work_group_size": None
        }

    def _fetch_results(self,
                       connection: Connection, model_path: str,
                       input_size_list: List[List[int]], flags) -> InferenceResult:
        result_str = self._launch_benchmark(connection, model_path, flags)
        if flags.get("kernel_path") is not None:
            connection.pull(flags["kernel_path"], ".")

        profiling_details = {
            "gpu_freq":  rfind_assign_int(result_str, "gpu_freq")
        }

        num_kernels = 0
        profiling_details["local_work_size"] = []
        while True:
            try:
                mark = "local_work_size[{}]".format(num_kernels)
                ans = rfind_assign(result_str, mark).strip()
                profiling_details["local_work_size"].append(ans)
            except:
                break
            num_kernels += 1

        for stage in ["write", "comp", "read"]:
            avg_ms = rfind_assign_float(result_str,  stage + '_avg_ms')
            std_ms = rfind_assign_float(result_str,  stage + '_std_ms')
            min_ms = rfind_assign_float(result_str,  stage + '_min_ms')
            max_ms = rfind_assign_float(result_str,  stage + '_max_ms')
            profiling_details[stage] = {
                "avg": avg_ms, "std": std_ms,
                "min": min_ms, "max": max_ms
            }

        if rfind_assign_int(result_str, 'count') >= 2:
            std_ms = rfind_assign_float(result_str, 'std') / 1e3
            avg_ms = rfind_assign_float(result_str, 'avg') / 1e3
        else:
            std_ms = 0
            avg_ms = rfind_assign_float(result_str, 'curr') / 1e3

        layerwise_info = []
        if flags.get("enable_op_profiling", False):
            for i in range(num_kernels):
                mark = "kernel[{}]".format(i)
                layerwise_info.append({
                    "name": mark,
                    "time": dict()
                })
                for metric in ["avg_ms", "std_ms"]:
                    tmp = rfind_assign(
                        result_str, "{}_{}".format(mark, metric)).strip()
                    layerwise_info[-1]["time"][metric] = tmp

        return InferenceResult(
            avg_ms=avg_ms, std_ms=std_ms,
            profiling_details=profiling_details,
            layerwise_info=None
        )
