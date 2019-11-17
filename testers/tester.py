import os
import datetime

import progressbar
import json

from .inference_sdks.inference_sdk import InferenceSdk
from .sampling.samplier import Sampler
from .utils import camel_case_to_snake_case, regularize_for_json


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
            'time': '{0:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()),
            'class_name': type(self).__name__,
            'inference_sdk': type(self.inference_sdk).__name__,
            'adb_device_id': self.adb_device_id,
            'sampler': type(self.sampler).__name__,
            'settings': self.settings,
            'benchmark_model_flags': self.benchmark_model_flags
        }
        with open('snapshot.json', 'w') as f:
            f.write(json.dumps(regularize_for_json(dic), indent=4))

    def run(self, settings, benchmark_model_flags):
        self.settings = settings
        self.benchmark_model_flags = benchmark_model_flags

        self._chdir_in()
        self._dump_snapshot()

        samples = self.sampler.get_samples()
        if self.settings.get('filter') is not None:
            samples = filter(self.settings['filter'], samples)
        samples = list(samples)

        bar = progressbar.ProgressBar(max_value=len(
            samples), redirect_stderr=True, redirect_stdout=True)
        csv_writer = CSVWriter()

        resumed = False
        bar.update(0)
        for i, sample in enumerate(samples):
            if self.settings.get('resume_from') is not None and not resumed:
                resumed = (sample == self.settings['resume_from'])
                continue

            results = self._test_sample(sample)
            csv_writer.update_data(self._get_csv_filename(sample), resumed,
                                   self.sampler.get_sample_titles() + self._get_metrics_titles(),
                                   sample + results)

            bar.update(i)

        self._chdir_out()
