from .tflite import Tflite
from .inference_sdk import InferenceResult
from .utils import rfind_assign_float, rfind_assign_int, rfind_assign
from utils.utils import adb_pull

import os


class TfliteModified(Tflite):
    @staticmethod
    def default_flags():
        return {
            "use_gpu": True,
            "work_group_size": None
        }

    def _fetch_results(self, adb_device_id: str, model_path: str, flags) -> InferenceResult:
        result_str = self._launch_benchmark(adb_device_id, model_path, flags)

        profiling_details = {
            "gpu_freq":  rfind_assign_int(result_str, "gpu_cur_freq")
        }

        i = 0
        while True:
            try:
                mark = "best_work_group[{}]".format(i)
                profiling_details[mark] = rfind_assign(result_str, mark).strip()
            except:
                break
            i += 1

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

        return InferenceResult(avg_ms=avg_ms, std_ms=std_ms, profiling_details=profiling_details)
