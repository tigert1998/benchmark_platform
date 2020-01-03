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
from utils.class_with_settings import ClassWithSettings


class Tester(ClassWithSettings):
    @staticmethod
    def default_settings():
        return {
            **ClassWithSettings.default_settings(),
            "adb_device_id": None,
            "inference_sdk": None,
            "sampler": None,
            "subdir": None,
            "resume_from": None
        }

    def __init__(self, settings):
        super().__init__(settings=settings)
        self.adb_device_id = self.settings["adb_device_id"]
        self.inference_sdk = self.settings["inference_sdk"]
        self.sampler = self.settings["sampler"]

    def _get_dir_name(self):
        if self.adb_device_id is not None:
            dir_name = "{}_{}".format(self.brief(), self.adb_device_id)
        else:
            dir_name = self.brief()
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

    def _test_sample(self, sample):
        return []

    def _dump_snapshot(self):
        snapshot = self.snapshot()

        if snapshot["adb_device_id"] is None:
            snapshot.pop("adb_device_id")
        else:
            snapshot["adb_device_id"] = inquire_adb_device(self.adb_device_id)

        snapshot["time"] = '{0:%Y-%m-%d %H:%M:%S}'.format(
            datetime.datetime.now())
        snapshot["benchmark_model_flags"] = self.inference_sdk.flags(
            self.benchmark_model_flags)
        snapshot["remark"] = ""

        with open('snapshot.json', 'w') as f:
            f.write(json.dumps(regularize_for_json(snapshot), indent=4))

    def run(self, benchmark_model_flags):
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

            results = self._test_sample(sample)
            sample_dic = {
                key: value for key, value in zip(self.sampler.get_sample_titles(), sample)
            }
            data = {
                **sample_dic,
                **results
            }

            csv_writer.update_data(
                self._get_csv_filename(sample), data, resumed)

            bar.update(i + 1)
            print()

        self._chdir_out()
