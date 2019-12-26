from .test_dwconv import TestDwconv
from .test_conv import TestConv
from testers.tester import Tester
from testers.inference_sdks.inference_sdk import InferenceResult


class TestWithTfliteModified:
    @staticmethod
    def _get_metrics_titles():
        titles = ["latency_ms", "std_ms"]
        for stage in ["write", "comp", "read"]:
            for metric in ["avg", "std"]:
                titles.append("{}_{}_ms".format(stage, metric))
        titles.append("gpu_freq")
        titles.append("best_work_groups")
        return titles

    @staticmethod
    def _postprocess_inference_results(results: InferenceResult):
        data = [results.avg_ms, results.std_ms]

        for stage in ["write", "comp", "read"]:
            for metric in ["avg", "std"]:
                data.append(results.profiling_details[stage][metric])
        data.append(results.profiling_details["gpu_freq"])

        i = 0
        while True:
            mark = "best_work_group[{}]".format(i)
            if results.profiling_details.get(mark) is not None:
                data.append(results.profiling_details.get(mark))
            else:
                break
            i += 1

        return data


def _generate_tester_with_tflite_modified(TestBaseClass):
    assert issubclass(TestBaseClass, Tester)

    class NewClass(TestWithTfliteModified, TestBaseClass):
        def _test_sample(self, sample):
            model_path = self._generate_model(sample)
            results = self.inference_sdk.fetch_results(
                self.adb_device_id, model_path, self.benchmark_model_flags)
            return self._postprocess_inference_results(results)

    NewClass.__name__ = TestBaseClass.__name__ + "WithTfliteModified"

    return NewClass


TestConvWithTfliteModified = _generate_tester_with_tflite_modified(TestConv)
TestDwconvWithTfliteModified = _generate_tester_with_tflite_modified(
    TestDwconv)
