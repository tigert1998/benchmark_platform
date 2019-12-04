import shutil
from collections import namedtuple

from ..class_with_settings import ClassWithSettings

InferenceResult = namedtuple(
    "InferenceResult", ["avg_ms", "std_ms", "profiling_details"])


class InferenceSdk(ClassWithSettings):
    @staticmethod
    def default_flags():
        return {}

    @classmethod
    def flags(cls, benchmark_model_flags):
        return {
            **cls.default_flags(),
            **benchmark_model_flags
        }

    def generate_model(self, path: str, inputs, outputs):
        """Generates a model to path without extension
        Args:
            path: model path without extension
            inputs: input tensors
            outputs: output tensors
        """
        pass

    def fetch_results(self, adb_device_id: str, model_path: str, benchmark_model_flags) -> InferenceResult:
        return self._fetch_results(adb_device_id, model_path, self.flags(benchmark_model_flags))

    def _fetch_results(self, adb_device_id: str, model_path: str, benchmark_model_flags) -> InferenceResult:
        """push model to an android device and fetch results
        Args:
            adb_device_id: adb device ID
            model_path: model path without extension
            flags: Flag dict for various kinds of benchmark_model tools

        Returns:
            InferenceResult
        """
        pass
