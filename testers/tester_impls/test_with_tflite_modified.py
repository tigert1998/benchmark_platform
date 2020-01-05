from .test_dwconv import TestDwconv
from .test_conv import TestConv
from testers.tester import Tester
from testers.inference_sdks.inference_sdk import InferenceResult


class TestWithTfliteModified:
    @staticmethod
    def _postprocess_inference_results(results: InferenceResult):
        data = {
            "latency_ms": results.avg_ms,
            "std_ms": results.std_ms
        }

        for stage in ["write", "comp", "read"]:
            for metric in ["avg", "std"]:
                metric_title = "{}_{}".format(stage, metric)
                data[metric_title] = results.profiling_details[stage][metric]

        data["gpu_freq"] = results.profiling_details["gpu_freq"]

        local_work_size = results.profiling_details["local_work_size"]
        for i in range(len(local_work_size)):
            data["local_work_size[{}]".format(i)] = local_work_size[i]

        return data


def _generate_tester_with_tflite_modified(TestBaseClass):
    assert issubclass(TestBaseClass, Tester)

    class NewClass(TestWithTfliteModified, TestBaseClass):
        def _test_sample(self, sample):
            model_path, input_size_list = self._generate_model(sample)
            results = self.inference_sdk.fetch_results(
                self.adb_device_id, model_path, input_size_list, self.benchmark_model_flags)
            return self._postprocess_inference_results(results)

    NewClass.__name__ = TestBaseClass.__name__ + "WithTfliteModified"

    return NewClass


TestConvWithTfliteModified = _generate_tester_with_tflite_modified(TestConv)
TestDwconvWithTfliteModified = _generate_tester_with_tflite_modified(
    TestDwconv)
