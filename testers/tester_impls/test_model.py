from testers.tester import Tester
from utils.utils import rm_ext

import tensorflow as tf
from .utils import append_layerwise_info


class TestModel(Tester):
    def _test_sample(self, sample):
        model_path = sample[0]
        return self.inference_sdk.fetch_results(
            self.connection, rm_ext(model_path), None, self.benchmark_model_flags)
