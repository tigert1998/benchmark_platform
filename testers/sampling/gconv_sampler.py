from .sampler import Sampler
from .utils import sparse_channels_from_imsize, available_imsizes

import itertools


class GconvSampler(Sampler):
    @staticmethod
    def get_sample_titles():
        return [
            "op", "input_imsize", "current_cin", "current_cout",
            "num_groups",  "stride", "kernel_size"
        ]

    def _get_samples_without_filter(self):
        for imsize in available_imsizes():
            for cin in sparse_channels_from_imsize(imsize):
                for num_groups, stride, ksize in itertools.product(
                        [2, 3, 4, 8], [1, 2], [1, 3, 5, 7]):
                    if ksize > imsize or cin % num_groups != 0:
                        continue
                    yield ["GroupedConv", imsize, cin, cin, num_groups, stride, ksize]
