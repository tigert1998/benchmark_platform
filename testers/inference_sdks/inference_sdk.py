import shutil
from collections import namedtuple

InferenceResult = namedtuple(
    "InferenceResult", ["avg_ms", "std_ms", "op_profiling"])


class InferenceSdk:
    @staticmethod
    def generate_model(path: str, inputs, outputs):
        """Generates a model to path without extension
        Args:
            path: model path without extension
            inputs: input tensors
            outputs: output tensors
        """
        pass

    @staticmethod
    def fetch_results(adb_device_id: str, model_path: str, flags) -> InferenceResult:
        """push model to an android device and fetch results
        Args:
            adb_device_id: adb device ID
            model_path: model path without extension
            flags: Flag dict for various kinds of benchmark_model tools

        Returns:
            InferenceResult
        """
        pass
