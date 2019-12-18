from .sampler import Sampler


class ModelSampler(Sampler):
    @staticmethod
    def default_settings():
        return {
            **Sampler.default_settings(),
            "model_paths": []
        }

    @staticmethod
    def get_sample_titles():
        return ["model_path"]

    def _get_samples_without_filter(self):
        for model_path in self.settings["model_paths"]:
            yield [model_path]
