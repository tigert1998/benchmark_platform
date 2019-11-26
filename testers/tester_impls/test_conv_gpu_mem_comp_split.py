from testers.tester_impls.test_conv import TestConv

import tensorflow as tf
import os


class TestConvGpuMemCompSplit(TestConv):
    @staticmethod
    def _get_metrics_titles():
        titles = ["latency_ms", "std_ms"]
        for stage in ["write", "comp", "read"]:
            for metric in ["avg", "std"]:
                titles.append("{}_{}_ms".format(stage, metric))
        titles.append("gpu_freq")
        return titles

    def _test_sample(self, sample):
        self._generate_model(sample)

        results = self.inference_sdk.fetch_results(
            self.adb_device_id, "model", self.benchmark_model_flags)

        data = [results.avg_ms, results.std_ms]

        for stage in ["write", "comp", "read"]:
            for metric in ["avg", "std"]:
                data.append(results.profiling_details[stage][metric])
        data.append(results.profiling_details["gpu_freq"])

        os.replace("kernel.cl", '_'.join(map(str, sample[2: 5])) + ".cl")

        return data
