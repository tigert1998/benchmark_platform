import os

import progressbar

from .inference_sdks.inference_sdk import InferenceSdk
from .sampling.samplier import Sampler
from .utils import camel_case_to_snake_case


class Tester:
    def __init__(self, adb_device_id: str,
                 inference_sdk: InferenceSdk, sampler: Sampler):
        self.adb_device_id = adb_device_id
        self.inference_sdk = inference_sdk
        self.sampler = sampler

    def get_dir_name(self):
        return "{}_{}_{}".format(
            camel_case_to_snake_case(type(self).__name__),
            camel_case_to_snake_case(type(self.inference_sdk).__name__),
            self.adb_device_id)

    def _chdir_in(self):
        dir_name = self.get_dir_name()
        if not os.path.isdir(dir_name):
            if os.path.exists(dir_name):
                print("os.path.exists(\"{}\")".format(dir_name))
                exit()
            else:
                os.mkdir(dir_name)
        os.chdir(dir_name)

    @staticmethod
    def _chdir_out():
        os.chdir("..")

    def test_sample(self, sample):
        pass

    def run(self, settings, benchmark_model_flags):
        self.settings = settings
        self.benchmark_model_flags = benchmark_model_flags

        self._chdir_in()

        samples = self.sampler.get_samples()
        if self.settings.get('filter') is not None:
            samples = filter(self.settings['filter'], samples)
        samples = list(samples)

        bar = progressbar.ProgressBar(max_value=len(
            samples), redirect_stderr=True, redirect_stdout=True)

        resumed = False
        bar.update(0)
        for i, sample in enumerate(samples):
            if self.settings['resume_from'] is not None and not resumed:
                resumed = (sample == self.settings['resume_from'])
                continue
            self.test_sample(sample)
            bar.update(i)

        self._chdir_out()
