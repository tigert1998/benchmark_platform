from .sampler import Sampler
from .utils import sparse_channels_from_imsize, available_imsizes

import itertools


class MbnetV2BlockSampler(Sampler):
    @staticmethod
    def get_sample_titles():
        return [
            "block", "input_imsize", "current_cin", "current_cout", "stride", "kernel_size"
        ]

    def _get_samples_without_filter(self):
        for imsize in available_imsizes():
            for cin in sparse_channels_from_imsize(imsize):
                for stride, ksize in itertools.product(
                        [1, 2], [3, 5, 7]):
                    if ksize > imsize:
                        continue
                    cout = stride * cin
                    yield ["MobileNetV2Block", imsize, cin, cout, stride, ksize]
