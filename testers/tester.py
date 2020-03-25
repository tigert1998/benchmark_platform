import os
import datetime

import progressbar
import json

from .inference_sdks.inference_sdk import InferenceSdk, InferenceResult
from .sampling.sampler import Sampler
from utils.utils import\
    camel_case_to_snake_case,\
    regularize_for_json

from utils.csv_writer import CSVWriter
from utils.class_with_settings import ClassWithSettings
from utils.connection import Connection


class Tester(ClassWithSettings):
    @staticmethod
    def default_settings():
        return {
            **ClassWithSettings.default_settings(),
            "connection": None,
            "inference_sdk": None,
            "sampler": None,
            "subdir": None,
            "dirname": None,
            "resume_from": None
        }

    def __init__(self, settings):
        super().__init__(settings=settings)
        self.connection: Connection = self.settings["connection"]
        self.inference_sdk = self.settings["inference_sdk"]
        self.sampler = self.settings["sampler"]

    def _get_dir_name(self):
        if self.settings.get("dirname") is not None:
            dir_name = self.settings["dirname"]
        else:
            dir_name = self.brief()
        if self.settings.get("subdir") is not None:
            dir_name += "/" + self.settings.get("subdir")
        return dir_name

    def _chdir_in(self):
        self.cwd = os.getcwd()
        dir_name = "test_results/{}".format(self._get_dir_name())
        if not os.path.isdir(dir_name):
            if os.path.exists(dir_name):
                print("os.path.exists(\"{}\")".format(dir_name))
                exit()
            else:
                os.makedirs(dir_name)
        os.chdir(dir_name)

    def _chdir_out(self):
        os.chdir(self.cwd)

    def _get_csv_filename(self, sample):
        return "data.csv"

    def _test_sample(self, sample) -> InferenceResult:
        ...

    def _process_inference_result(self, result: InferenceResult):
        ret = {}
        if result.avg_ms is not None:
            ret = {
                **ret,
                "latency_ms": result.avg_ms,
                "std_ms": result.std_ms
            }

        # layerwise info
        if result.layerwise_info is not None:
            for dic in result.layerwise_info:
                ret[dic["name"] + "_avg_ms"] = dic["time"]["avg_ms"]
                ret[dic["name"] + "_std_ms"] = dic["time"]["std_ms"]

        # profiling details
        if result.profiling_details is not None:
            for stage in ["write", "comp", "read"]:
                if result.profiling_details.get(stage) is None:
                    continue
                for metric in ["avg", "std"]:
                    metric_title = "{}_{}_ms".format(stage, metric)
                    ret[metric_title] = result.profiling_details[stage][metric]

            if result.profiling_details.get("gpu_freq") is not None:
                ret["gpu_freq"] = result.profiling_details["gpu_freq"]

            if result.profiling_details.get("local_work_size") is not None:
                local_work_size = result.profiling_details["local_work_size"]
                for i in range(len(local_work_size)):
                    ret["local_work_size[{}]".format(i)] = local_work_size[i]

            if result.profiling_details.get("flops") is not None:
                ret["flops"] = result.profiling_details["flops"]
                ret["mac"] = result.profiling_details["mac"]

        return ret

    def _dump_snapshot(self):
        snapshot = self.snapshot()

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

            sample_dic = self.sampler.get_sample_dict(sample)
            result = self._test_sample(sample)
            result = self._process_inference_result(result)

            data = {
                **sample_dic,
                **result
            }
            csv_writer.update_data(
                self._get_csv_filename(sample), data, resumed)

            bar.update(i + 1)
            print()

        self._chdir_out()
