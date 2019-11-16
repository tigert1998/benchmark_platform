import shutil
from collections import namedtuple

InferenceResult = namedtuple("InferenceResult", ["avg_ms", "std_ms", "op_profiling"])

class InferenceSdk:
    @staticmethod
    def generate_model(path: str, inputs, outputs):
        pass

    @staticmethod
    def fetch_results(adb_device_id, flags) -> InferenceResult:
        pass