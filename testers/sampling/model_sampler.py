from .sampler import Sampler


class ModelSampler(Sampler):
    @staticmethod
    def default_settings():
        return {
            **Sampler.default_settings(),
            "model_details": []
        }

    def snapshot(self):
        ret = super().snapshot()
        ret.pop("model_details")
        ret["model_paths"] = []
        for model_detail in self.settings["model_details"]:
            ret["model_paths"].append(model_detail.model_path)
        return ret

    @staticmethod
    def get_sample_titles():
        return ["model_path"]

    @staticmethod
    def _get_serializable_sample(sample):
        return [sample[0].model_path]

    def _get_samples_without_filter(self):
        for model_detail in self.settings["model_details"]:
            yield [model_detail]
