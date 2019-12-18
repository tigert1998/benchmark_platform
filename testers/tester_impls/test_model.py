from testers.tester import Tester
from utils.utils import rm_ext

import tensorflow as tf


class TestModel(Tester):
    @staticmethod
    def _get_metrics_titles():
        return ["latency_ms", "std_ms"]

    def _test_sample(self, sample):
        model_path = sample[0]
        results = self.inference_sdk.fetch_results(
            self.adb_device_id, rm_ext(model_path), self.benchmark_model_flags)
        return [results.avg_ms, results.std_ms]
