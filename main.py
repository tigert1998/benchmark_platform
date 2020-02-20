from glob import glob
import numpy as np

from testers.tester_impls.test_conv import TestConv
from testers.tester_impls.test_dwconv import TestDwconv
from testers.sampling.conv_sampler import SimpleConvSampler
from testers.sampling.dwconv_sampler import SimpleDwconvSampler

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
            "model_paths":
            list(map(lambda x: x.model_path,
                     get_model_details(None, "tflite", ["float16"])))
        })
    })

    tester.run(benchmark_model_flags={
        "num_runs": 30,
        "use_gpu": False,
        "gpu_precision_loss_allowed": False
    })


def accuracy_test_rknn():
    from accuracy_tester.accuracy_tester import AccuracyTester
    from accuracy_tester.accuracy_evaluators.rknn import Rknn

    tester = AccuracyTester({
        "zip_size": 50000,
        "dataset_size": 100,
        "model_details": get_model_details(None, "rknn", [""]),
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
        "model_details": get_model_details(["shufflenet"], "pb", [""]),
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

    tester = AccuracyTester({
        "zip_size": 50000,
        "dataset_size": 100,
        "model_details": get_model_details(["shufflenet"], "pb", [""]),
        "data_preparer": AndroidDataPreparer({
            "labels_path": "C:/Users/tigertang/Projects/dataset/val_labels.txt",
            "validation_set_path": "C:/Users/tigertang/Projects/dataset/validation",
            "skip_dataset_preparation": True,
            "skip_models_preparation": True,

            "connection": Adb("2e98c8a5", False),
        }),
        "accuracy_evaluator": Tflite({
            "connection": Adb("2e98c8a5", False),

            # on guest
            "imagenet_accuracy_eval_path": "/data/local/tmp/tf-r2.1-60afa4e/imagenet_accuracy_eval",
            "imagenet_accuracy_eval_flags": {
            },
        })
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
        "sampler": SimpleDwconvSampler({
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
        "connection": Ssh("zhongrg@zhongrg-All-Series"),
        "inference_sdk": Tpu(),
        "sampler": SimpleConvSampler({}),
        "resume_from": ["", "Conv", 7, 640, 816, "", "", 1, 3]
    })
    tester.run({})


def layer_latency_test_rknn():
    from testers.inference_sdks.rknn import Rknn

    # tester = TestDwconv({
    #     "inference_sdk": Rknn({}),
    #     "sampler": SimpleDwconvSampler({}),
    #     "resume_from": ["", "DWConv", 224, 424, 424, "", "", 1, 3]
    # })
    # tester.run({})

    tester = TestConv({
        "connection": Adb("TD033101190100171", False),
        "inference_sdk": Rknn({
            "rknn_target": None,
        }),
        "sampler": SimpleConvSampler({}),
        "resume_from": ["", "Conv", 7, 640, 344, "", "", 2, 3]
    })
    tester.run({})


if __name__ == '__main__':
    accuracy_test_pb()
