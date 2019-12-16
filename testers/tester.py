import os
import datetime

import progressbar
import json

from .inference_sdks.inference_sdk import InferenceSdk
from .sampling.sampler import Sampler
from utils.utils import\
    camel_case_to_snake_case,\
    regularize_for_json,\
    adb_shell_su, adb_shell, shell_with_script,\
    inquire_adb_device
from utils.csv_writer import CSVWriter


class Tester:
    def __init__(self, adb_device_id: str,
                 inference_sdk: InferenceSdk, sampler: Sampler):
        self.adb_device_id = adb_device_id
        self.inference_sdk = inference_sdk
        self.sampler = sampler

    def _get_dir_name(self):
        dir_name = "{}_{}_{}".format(
            camel_case_to_snake_case(type(self).__name__),
            camel_case_to_snake_case(type(self.inference_sdk).__name__),
            self.adb_device_id)
        if self.settings.get("subdir") is not None:
            dir_name += "/" + self.settings.get("subdir")
        return dir_name

    def _chdir_in(self):
        dir_name = self._get_dir_name()
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

    def _get_csv_filename(self, sample):
        return "data.csv"

    @staticmethod
    def _get_metrics_titles():
        return []

    def _test_sample(self, sample):
        return []

    def _dump_snapshot(self):
        dic = {
            'class_name': type(self).__name__,
            'time': '{0:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()),
            'inference_sdk': self.inference_sdk.settings,
            'sampler': self.sampler.settings,
            'adb_device': inquire_adb_device(self.adb_device_id),
            'settings': self.settings,
            'benchmark_model_flags': self.inference_sdk.flags(self.benchmark_model_flags),
            "remark": ""
        }
        with open('snapshot.json', 'w') as f:
            f.write(json.dumps(regularize_for_json(dic), indent=4))

    def run(self, settings, benchmark_model_flags):
        self.settings = settings
        self.benchmark_model_flags = benchmark_model_flags

        self._chdir_in()
        self._dump_snapshot()

        samples = list(self.sampler.get_samples())

        bar = progressbar.ProgressBar(
            max_value=len(samples),
            redirect_stderr=False,
            redirect_stdout=False)
        csv_writer = CSVWriter()

        resumed = False
        bar.update(0)
        print()

        for i, sample in enumerate(samples):
            if self.settings.get('resume_from') is not None and not resumed:
                resumed = (sample == self.settings['resume_from'])
                continue

            self.preprocess()

            results = self._test_sample(sample)
            titles = self.sampler.get_sample_titles() + self._get_metrics_titles()
            data = sample + results

            titles, data = self.postprocess(titles, data)

            csv_writer.update_data(self._get_csv_filename(sample),
                                   titles, data, resumed)

            bar.update(i + 1)
            print()

        self._chdir_out()

    # preprocessing/postprocessing

    def _push_to_max_freq(self):
        adb_shell_su(self.adb_device_id,
                     shell_with_script("push_to_max_freq", self.settings.get("mkshrc", "/system/etc/mkshrc")))

    def _pull_gpu_cur_freq(self):
        if not self.settings.get("pull_gpu_cur_freq", False):
            return
        return int(adb_shell_su(
            self.adb_device_id,
            shell_with_script("cat $GPUDIR/cur_freq", self.settings.get("mkshrc", "/system/etc/mkshrc"))))

    def preprocess(self):
        if self.settings.get("push_to_max_freq", False):
            self._push_to_max_freq()

    def postprocess(self, titles, data):
        if self.settings.get("pull_gpu_cur_freq", False):
            titles.append("gpu_freq")
            data.append(self._pull_gpu_cur_freq())
        return titles, data
