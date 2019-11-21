import os
import datetime

import progressbar
import json

from .inference_sdks.inference_sdk import InferenceSdk
from .sampling.sampler import Sampler
from .utils import camel_case_to_snake_case, regularize_for_json, adb_shell_su, adb_shell


class CSVWriter:
    def __init__(self):
        self.fd = None
        self.previous_filename = None

    def _write_titles(self, titles):
        self.fd.write(','.join(titles) + '\n')
        self.fd.flush()

    def _write_data(self, data):
        self.fd.write(','.join(map(str, data)) + '\n')
        self.fd.flush()

    def _close(self):
        if self.fd is not None:
            self.fd.close()

    def update_data(self, filename, is_resume, titles, data):
        if self.previous_filename is None:
            if is_resume:
                self.fd = open(filename, 'a')
            else:
                self.fd = open(filename, 'w')
                self._write_titles(titles)
            self._write_data(data)
        elif self.previous_filename != filename:
            self._close()
            self.fd = open(filename, 'w')
            self._write_titles(titles)
            self._write_data(data)
        else:
            self._write_data(data)
        self.previous_filename = filename

    def __del__(self):
        self._close()


class Tester:
    def __init__(self, adb_device_id: str,
                 inference_sdk: InferenceSdk, sampler: Sampler):
        self.adb_device_id = adb_device_id
        self.inference_sdk = inference_sdk
        self.sampler = sampler

    def _get_dir_name(self):
        return "{}_{}_{}".format(
            camel_case_to_snake_case(type(self).__name__),
            camel_case_to_snake_case(type(self.inference_sdk).__name__),
            self.adb_device_id)

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
            'adb_device_id': self.adb_device_id,
            'settings': self.settings,
            'benchmark_model_flags': self.benchmark_model_flags
        }
        with open('snapshot.json', 'w') as f:
            f.write(json.dumps(regularize_for_json(dic), indent=4))

    def _push_to_max_freq(self):
        if not self.settings.get("push_to_max_freq", False):
            return
        adb_shell_su(self.adb_device_id, "source {} && push_to_max_freq".format(
            self.settings.get("mkshrc", "/system/etc/mkshrc")))

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
        for i, sample in enumerate(samples):
            if self.settings.get('resume_from') is not None and not resumed:
                resumed = (sample == self.settings['resume_from'])
                continue

            self._push_to_max_freq()

            results = self._test_sample(sample)
            csv_writer.update_data(self._get_csv_filename(sample), resumed,
                                   self.sampler.get_sample_titles() + self._get_metrics_titles(),
                                   sample + results)

            bar.update(i + 1)

        self._chdir_out()
