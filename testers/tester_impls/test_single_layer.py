from testers.tester import Tester

import tensorflow as tf

from typing import Tuple, List


class TestSingleLayer(Tester):
    def _generate_model(self, sample) -> Tuple[str, List[List[int]]]:
        """generate a single-layer model from sample
        Returns:
            model_path: model path without extension
            input_size_list: inputs shapes of the model
        """
        ...

    def _test_sample(self, sample):
        model_path, input_size_list = self._generate_model(sample)
        return self.inference_sdk.fetch_results(
            self.connection, model_path, input_size_list, self.benchmark_model_flags)
