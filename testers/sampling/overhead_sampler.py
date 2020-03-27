from .sampler import Sampler

import itertools


class OverheadSampler(Sampler):
    @staticmethod
    def get_sample_titles():
        return ["imsize", "cin"]

    def _get_samples_without_filter(self):
        yield [28, 320]
        for i in [16, 100]:
            yield [28, i]
