from .tflite import Tflite
from .inference_sdk import InferenceResult
from .utils import rfind_assign_float


class TfliteGpuMemCompSplit(Tflite):
    @staticmethod
    def default_flags():
        return {
            "use_gpu": True
        }

    def _fetch_results(self, adb_device_id: str, model_path: str, flags) -> InferenceResult:
        result_str = self._launch_benchmark(adb_device_id, model_path, flags)

        profiling_details = {}
        for stage in ["write", "comp", "read"]:
            avg_ms = rfind_assign_float(result_str,  stage + '_avg_ms')
            std_ms = rfind_assign_float(result_str,  stage + '_std_ms')
            profiling_details[stage] = {"avg": avg_ms, "std": std_ms}

        if flags.get("num_runs") is None or flags.get("num_runs") >= 2:
            std_ms = rfind_assign_float(result_str, 'std') / 1e3
            avg_ms = rfind_assign_float(result_str, 'avg') / 1e3
        else:
            std_ms = 0
            avg_ms = rfind_assign_float(result_str, 'curr') / 1e3

        return InferenceResult(avg_ms=avg_ms, std_ms=std_ms, profiling_details=profiling_details)
