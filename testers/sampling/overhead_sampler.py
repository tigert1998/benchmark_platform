from .sampler import Sampler

import itertools
from .utils import sparse_channels_from_imsize, available_imsizes


class OverheadSampler(Sampler):
    @staticmethod
    def get_sample_titles():
        return ["imsize", "cin"]

    def _get_samples_without_filter(self):
        for imsize in available_imsizes():
            for cin in sparse_channels_from_imsize(imsize):
                yield [imsize, cin]
