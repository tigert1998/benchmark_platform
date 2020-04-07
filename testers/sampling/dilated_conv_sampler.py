from .sampler import Sampler
from .utils import sparse_channels_from_imsize, available_imsizes

import itertools


class DilatedConvSampler(Sampler):
    @staticmethod
    def get_sample_titles():
        return [
            "op", "input_imsize", "current_cin", "current_cout",
            "dilation",  "stride", "kernel_size"
        ]

    def _get_samples_without_filter(self):
        for imsize in available_imsizes():
            for cin in sparse_channels_from_imsize(imsize):
                for dilation, stride, ksize in itertools.product([2, 4], [1, 2], [3, 5, 7]):
                    receptive_field = (ksize - 1) * dilation + 1
                    if imsize < receptive_field or imsize < stride:
                        continue
                    # strides > 1 not supported in conjunction with dilation_rate > 1
                    if stride > 1 and dilation > 1:
                        continue
                    yield ["DilatedConv", imsize, cin, cin, dilation, stride, ksize]
