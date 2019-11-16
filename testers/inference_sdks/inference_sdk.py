import shutil
from collections import namedtuple

InferenceResult = namedtuple(
    "InferenceResult", ["avg_ms", "std_ms", "op_profiling"])


class InferenceSdk:
    @staticmethod
    def generate_model(path: str, inputs, outputs):
        pass

    @staticmethod
    def fetch_results(adb_device_id, flags) -> InferenceResult:
        """push model to an android device and fetch results
        Args:
            adb_device_id: adb device ID
            flags: Flag dict for various kinds of benchmark_model tools

        Returns:
            InferenceResult
        """
        pass
