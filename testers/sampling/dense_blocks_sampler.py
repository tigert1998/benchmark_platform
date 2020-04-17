from .sampler import Sampler
from .utils import sparse_channels_from_imsize, available_imsizes

import itertools


class DenseBlockSampler(Sampler):
    @staticmethod
    def get_sample_titles():
        return [
            "block", "input_imsize", "current_cin", "growth_rate", "num_layers", "kernel_size"
        ]

    def _get_samples_without_filter(self):
        for imsize in available_imsizes():
            for cin in sparse_channels_from_imsize(imsize):
                for ksize in [3, 5, 7]:
                    if imsize < ksize:
                        continue
                    for num_layers in [1, 2]:
                        yield ["DenseBlock", imsize, cin, 32, num_layers, ksize]
