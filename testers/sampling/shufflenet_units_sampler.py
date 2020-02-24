from .sampler import Sampler
from .utils import sparse_channels_from_imsize, available_imsizes

import itertools


class ShufflenetV1UnitSampler(Sampler):
    @staticmethod
    def get_sample_titles():
        return [
            "block", "input_imsize", "current_cin", "current_cout",
            "num_groups", "mid_channels", "stride", "kernel_size"
        ]

    def _get_samples_without_filter(self):
        for imsize in available_imsizes():
            for cin in sparse_channels_from_imsize(imsize):
                for num_groups, stride, ksize in itertools.product(
                    [2, 3, 4, 8], [1, 2], [3, 5, 7]
                ):
                    if ksize > imsize:
                        continue
                    if cin % num_groups != 0:
                        continue
                    if stride == 1:
                        cout = cin
                    else:
                        cout = 2 * cin
                    mid_channels = cout // 4
                    yield [
                        "ShufflenetV1Unit", imsize, cin, cout,
                        num_groups, mid_channels, stride, ksize
                    ]


class ShufflenetV2UnitSampler(Sampler):
    @staticmethod
    def get_sample_titles():
        return [
            "block", "input_imsize", "current_cin", "stride", "kernel_size"
        ]

    def _get_samples_without_filter(self):
        for imsize in available_imsizes():
            for cin in sparse_channels_from_imsize(imsize):
                for stride, ksize in itertools.product(
                    [1, 2], [3, 5, 7]
                ):
                    if ksize > imsize:
                        continue
                    yield [
                        "ShufflenetV2Unit", imsize, cin, stride, ksize
                    ]
