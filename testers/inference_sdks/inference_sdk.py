import shutil
from collections import namedtuple

from utils.class_with_settings import ClassWithSettings
from utils.connection import Connection

InferenceResult = namedtuple(
    "InferenceResult", ["avg_ms", "std_ms", "profiling_details", "layerwise_info"])


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

    def fetch_results(self, connection: Connection, model_path: str, input_size_list, benchmark_model_flags) -> InferenceResult:
        return self._fetch_results(connection, model_path, input_size_list, self.flags(benchmark_model_flags))

    def _fetch_results(self, connection: Connection, model_path: str, input_size_list, benchmark_model_flags) -> InferenceResult:
        """push model to an android device and fetch results
        Args:
            connection: Connection
            model_path: model path without extension
            input_size_list: input_shape of the model
            flags: Flag dict for various kinds of benchmark_model tools

        Returns:
            InferenceResult
        """
        pass
