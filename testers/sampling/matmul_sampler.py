from .sampler import Sampler

import itertools


class MatmulSampler(Sampler):
    @staticmethod
    def get_sample_titles():
        return ["n"]

    def _get_samples_without_filter(self):
        for i in range(2, 101):
            yield [i]
