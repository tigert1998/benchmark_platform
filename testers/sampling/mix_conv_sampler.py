from .sampler import Sampler
from .utils import sparse_channels_from_imsize, available_imsizes

import itertools


class MixConvSampler(Sampler):
    @staticmethod
    def get_sample_titles():
        return [
            "block", "input_imsize", "current_cin", "current_cout", "num_groups", "stride"
        ]

    def _get_samples_without_filter(self):
        for imsize in available_imsizes():
            for cin in sparse_channels_from_imsize(imsize):
                for num_groups in range(2, 6):
                    max_ksize = num_groups * 2 + 1
                    if max_ksize > imsize:
                        continue
                    if cin % num_groups != 0:
                        continue
                    for stride in [1, 2]:
                        if imsize < stride:
                            continue
                        yield ["MixConv", imsize, cin, cin, num_groups, stride]
