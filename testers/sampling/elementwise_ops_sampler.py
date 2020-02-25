from .sampler import Sampler
from .utils import sparse_channels_from_imsize, available_imsizes

import itertools


class AddSampler(Sampler):
    @staticmethod
    def get_sample_titles():
        return [
            "op", "input_imsize", "current_cin"
        ]

    def _get_samples_without_filter(self):
        for imsize in available_imsizes():
            for cin in sparse_channels_from_imsize(imsize):
                yield ["Add", imsize, cin]


class ConcatSampler(Sampler):
    @staticmethod
    def get_sample_titles():
        return [
            "op", "input_imsize", "first_input_cin", "second_input_cin"
        ]

    def _get_samples_without_filter(self):
        for imsize in available_imsizes():
            for cin in sparse_channels_from_imsize(imsize):
                for second_cin in sorted([cin, 32]):
                    yield ["Concat", imsize, cin, second_cin]


class GlobalPoolingSampler(Sampler):
    @staticmethod
    def get_sample_titles():
        return [
            "op", "input_imsize", "current_cin"
        ]

    def _get_samples_without_filter(self):
        for cin in [1024, 1280]:
            yield ["GlobalAveragePooling", 7, cin]


class ShuffleSampler(Sampler):
    @staticmethod
    def get_sample_titles():
        return [
            "op", "input_imsize", "current_cin", "num_groups"
        ]

    def _get_samples_without_filter(self):
        for imsize in available_imsizes():
            for cin in sparse_channels_from_imsize(imsize):
                for num_groups in [2, 3, 4, 8]:
                    if cin % num_groups != 0:
                        continue
                    yield ["Shuffle", imsize, cin, num_groups]
