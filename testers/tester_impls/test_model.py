from testers.tester import Tester
from preprocess.model_archive import ModelDetail
from utils.utils import rm_ext

import tensorflow as tf
from .utils import append_layerwise_info


class TestModel(Tester):
    def _test_sample(self, sample):
        model_detail: ModelDetail = sample[0]
        imsize: int = model_detail.preprocess.preprocessor.imsize
        return self.inference_sdk.fetch_results(
            self.connection,
            rm_ext(model_detail.model_path),
            [[1, imsize, imsize, 3]],
            self.benchmark_model_flags
        )
