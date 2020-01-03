from testers.tester import Tester
from utils.utils import rm_ext

import tensorflow as tf
from .utils import append_layerwise_info


class TestModel(Tester):
    def _test_sample(self, sample):
        model_path = sample[0]
        results = self.inference_sdk.fetch_results(
            self.adb_device_id, rm_ext(model_path), None, self.benchmark_model_flags)
        return append_layerwise_info({
            "latency_ms": results.avg_ms,
            "std_ms": results.std_ms
        }, results.layerwise_info)
