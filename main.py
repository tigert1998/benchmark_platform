from glob import glob
import numpy as np

from testers.tester_impls.test_conv import TestConv
from testers.tester_impls.test_dwconv import TestDwconv
from testers.sampling.conv_sampler import ChannelExperimentConvSampler
from testers.sampling.dwconv_sampler import ChannelExperimentDwconvSampler

from accuracy_tester.data_preparers.android_data_preparer import AndroidDataPreparer
from accuracy_tester.data_preparers.data_preparer_def import DataPreparerDef

from utils.connection import Adb, Ssh, Connection

from preprocess.model_archive import get_model_details


def model_latency_test():
    from testers.tester_impls.test_model import TestModel
    from testers.inference_sdks.tflite import Tflite
    # from testers.inference_sdks.rknn import Rknn
    from testers.sampling.model_sampler import ModelSampler

    tester = TestModel(settings={
        "connection": Adb("2e98c8a5", False),
        "inference_sdk": Tflite({
            "benchmark_model_path": "/data/local/tmp/tf-r2.1-60afa4e/benchmark_model",
        }),
        "sampler": ModelSampler({
            "model_details":
                get_model_details(None, "tflite", [
                    "", "float16",
                ], "mobile_gpu")
        })
    })

    tester.run(benchmark_model_flags={
        "use_gpu": True
    })


def model_flops_test():
    from testers.tester_impls.test_model import TestModel
    from testers.inference_sdks.flops_calculator import FlopsCalculator
    from testers.sampling.model_sampler import ModelSampler

    tester = TestModel(settings={
        "connection": Connection(),
        "inference_sdk": FlopsCalculator({}),
        "sampler": ModelSampler({
            "model_details": get_model_details(None, "pb", ["patched"])
        })
    })

    tester.run(benchmark_model_flags={})


def accuracy_test_rknn():
    from accuracy_tester.accuracy_tester import AccuracyTester
    from accuracy_tester.accuracy_evaluators.rknn import Rknn

    tester = AccuracyTester({
        "dirname": "test_rknn",
        "zip_size": 50000,
        "dataset_size": 50000,
        "model_details": get_model_details(["inception_v4"], "rknn", [
            "dynamic_fixed_point_8", "dynamic_fixed_point_16"
        ]),
        "data_preparer": DataPreparerDef({
            "labels_path": "C:/Users/v-xiat/Downloads/playground/imagenet/val_labels.txt",
            "validation_set_path": "C:/Users/v-xiat/Downloads/playground/imagenet/validation",
            "skip_dataset_preparation": True,
            "skip_models_preparation": True,
        }),
        "accuracy_evaluator": Rknn({})
    })
    tester.run()


def accuracy_test_pb():
    from accuracy_tester.accuracy_tester import AccuracyTester
    from accuracy_tester.accuracy_evaluators.tf_evaluator import TfEvaluator

    tester = AccuracyTester({
        "zip_size": 50000,
        "dataset_size": 100,
        "model_details": get_model_details(None, "pb", ["patched"]),
        "data_preparer": DataPreparerDef({
            "labels_path": "C:/Users/tigertang/Projects/dataset/val_labels.txt",
            "validation_set_path": "C:/Users/tigertang/Projects/dataset/validation",
            "skip_dataset_preparation": True,
            "skip_models_preparation": True,
        }),
        "accuracy_evaluator": TfEvaluator({})
    })
    tester.run()


def accuracy_test_tflite():
    from accuracy_tester.accuracy_tester import AccuracyTester
    from accuracy_tester.accuracy_evaluators.tflite import Tflite
    from accuracy_tester.accuracy_evaluators.tpu import Tpu

    tester = AccuracyTester({
        "zip_size": 50000,
        "dataset_size": 100,
        "model_details": get_model_details(None, "tflite", ["edgetpu"], "edgetpu"),
        "data_preparer": DataPreparerDef({
            "labels_path": "/home/hanxiao/benchmarks/val.txt",
            "validation_set_path": "/home/hanxiao/benchmarks/imagenet_dataset",
            "skip_dataset_preparation": True,
            "skip_models_preparation": True,
        }),
        "accuracy_evaluator": Tpu({})
    })
    tester.run()


def layer_latency_test_tflite():
    from testers.inference_sdks.tflite_modified import TfliteModified
    from testers.inference_sdks.tflite import Tflite

    tester = TestDwconv({
        "connection": Adb("5e6fecf", True),
        "inference_sdk": TfliteModified({
            "benchmark_model_path": "/data/local/tmp/tf-r2.1-60afa4e/benchmark_model_modified",
        }),
        "sampler": ChannelExperimentDwconvSampler({
            # "filter": lambda sample: sample[2: 5] == [7, 960, 960]
        }),
        "resume_from": ["", "DWConv", 224, 64, 64, "", "", 2, 7]
    })
    tester.run({
        "use_gpu": True,
        "work_group_size": "",
        "tuning_type": "EXHAUSTIVE",
        "kernel_path": "/data/local/tmp/kernel.cl"
    })


def layer_latency_test_tpu():
    from testers.inference_sdks.tpu import Tpu

    tester = TestConv({
        "connection": Connection(),
        "inference_sdk": Tpu({
            "edgetpu_compiler_path": "/home/xiaohu/edgetpu/compiler/x86_64/edgetpu_compiler",
            "libedgetpu_path": "/home/xiaohu/edgetpu/libedgetpu/direct/k8/libedgetpu.so.1"
        }),
        "sampler": ChannelExperimentConvSampler({}),
        # "resume_from": ["", "Conv", 7, 160, 880, "", "", 1, 3]
    })
    tester.run({})


def layer_latency_test_rknn():
    from testers.inference_sdks.rknn import Rknn

    tester = TestConv({
        "connection": Adb("TD033101190100171", False),
        "inference_sdk": Rknn({
            "rknn_target": None,
            "quantization": "asymmetric_quantized-u8"
        }),
        "sampler": ChannelExperimentConvSampler({}),
        "resume_from": ["", "Conv", 7, 576, 640, "", "", 1, 3]
    })
    tester.run({})


if __name__ == '__main__':
    accuracy_test_tflite()
