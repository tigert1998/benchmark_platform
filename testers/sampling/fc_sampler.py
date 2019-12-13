from .sampler import Sampler


class FcSampler(Sampler):
    @staticmethod
    def get_sample_titles():
        return ["model", "op",  "current_cin", "current_cout",
                "original_cin", "original_cout"]

    def _get_samples_without_filter(self):
        original_cin = 1000
        for cin in range(int(0.2 * original_cin), 2 * original_cin):
            yield ["MobileNetV2", "FC", cin, original_cin, original_cin, original_cin]