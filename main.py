from glob import glob
from utils.preprocess import InceptionPreprocess, VggPreprocess
import numpy as np

from testers.tester_impls.test_conv import TestConv
from testers.tester_impls.test_dwconv import TestDwconv
from testers.sampling.conv_sampler import SimpleConvSampler
from testers.sampling.dwconv_sampler import SimpleDwconvSampler

from utils.connection import Adb, Ssh


def accuracy_test_rknn():
    from accuracy_tester.accuracy_tester import AccuracyTester
    from accuracy_tester.data_preparers.data_preparer_def import DataPreparerDef
    from accuracy_tester.accuracy_evaluators.rknn import Rknn

    tester = AccuracyTester({
        "zip_size": 50000,
        "model_paths": glob("C:/Users/v-xiat/Downloads/imagenet/models/efficientnet_b0_patched.rknn"),
        "data_preparer": DataPreparerDef({
            "labels_path": "C:/Users/v-xiat/Downloads/imagenet/val_labels.txt",
            "validation_set_path": "C:/Users/v-xiat/Downloads/imagenet/validation",
        }),
        "accuracy_evaluator": Rknn({
            "preprocess": lambda image: InceptionPreprocess.resize(image, 224),
            "index_to_label": lambda index: str(index + 1)
        })
    })
    tester.run()


def accuracy_test_tflite():
    from accuracy_tester.accuracy_tester import AccuracyTester
    from accuracy_tester.data_preparers.android_data_preparer import AndroidDataPreparer
    from accuracy_tester.accuracy_evaluators.tflite import Tflite

    tester = AccuracyTester({
        "zip_size": 500,
        "dirname": "test_efficientnet_b0",
        "model_paths": glob("C:/Users/v-xiat/repos/efficientnet_b1_int_quant.tflite"),
        "data_preparer": AndroidDataPreparer({
            "labels_path": "C:/Users/v-xiat/Downloads/imagenet/val_labels.txt",
            "validation_set_path": "C:/Users/v-xiat/Downloads/imagenet/validation",
            "connection": Adb("5e6fecf", False),
            "skip_dataset_preparation": True,
            "skip_models_preparation": False
        }),
        "accuracy_evaluator": Tflite({
            "eval_on_host": False,

            # on guest
            "connection": Adb("5e6fecf", False),
            "imagenet_accuracy_eval_path": "/data/local/tmp/tf-r2.1-60afa4e/imagenet_accuracy_eval",
            "imagenet_accuracy_eval_flags": {
                "num_images": 500,
            },

            # on guest
            "preprocess": lambda image: InceptionPreprocess.resize(image, 224),
            "index_to_label": lambda index: str(index + 1)
        })
    })
    tester.run()


def model_latency_test():
    from testers.tester_impls.test_model import TestModel
    from testers.inference_sdks.tflite import Tflite
    from testers.inference_sdks.rknn import Rknn
    from testers.sampling.model_sampler import ModelSampler

    tester = TestModel(settings={
        "connection": Adb("5e6fecf", False),
        "inference_sdk": Rknn({
            "rknn_target": "rk1808",
            "input_imsize": 224
        }),
        "sampler": ModelSampler({
            "model_paths": glob("C:/Users/v-xiat/Microsoft/Shihao Han (FA Talent) - ChannelNas/models/rknn/inception_v4/*.rknn")
        })
    })

    tester.run(benchmark_model_flags={
        "num_runs": 30,
        "use_gpu": False,
        "gpu_precision_loss_allowed": False
    })


def accuracy_test_pb():
    from accuracy_tester.accuracy_tester import AccuracyTester
    from accuracy_tester.data_preparers.data_preparer_def import DataPreparerDef
    from accuracy_tester.accuracy_evaluators.tf_evaluator import TfEvaluator

    tester = AccuracyTester({
        "zip_size": 50000,
        "model_paths": ["C:/Users/v-xiat/Microsoft/Shihao Han (FA Talent) - ChannelNas/models/pb/efficientnet_b0_patched.pb"],
        "data_preparer": DataPreparerDef({
            "labels_path": "C:/Users/v-xiat/Downloads/imagenet/val_labels.txt",
            "validation_set_path": "C:/Users/v-xiat/Downloads/imagenet/validation",
        }),
        "accuracy_evaluator": TfEvaluator({
            "preprocess": lambda image: InceptionPreprocess.preprocess(image, 224),
            "index_to_label": lambda index: str(index + 1)
        })
    })
    tester.run()


def layer_latency_test_tflite():
    from testers.inference_sdks.tflite_modified import TfliteModified
    from testers.inference_sdks.tflite import Tflite

    tester = TestConv({
        "connection": Adb("5e6fecf", False),
        "inference_sdk": Tflite({
            "benchmark_model_path": "/data/local/tmp/tf-r2.1-60afa4e/benchmark_model",
        }),
        "sampler": SimpleConvSampler({
            # "filter": lambda sample: sample[2: 5] == [7, 960, 960]
        }),
        "resume_from": ["", "Conv", 112, 102, 320, "", "", 2, 1]
    })
    tester.run({
        "use_gpu": False,
        "work_group_size": ""
    })


def layer_latency_test_tpu():
    from testers.inference_sdks.tpu import Tpu

    tester = TestConv({
        "connection": Ssh("zhongrg@zhongrg-All-Series"),
        "inference_sdk": Tpu(),
        "sampler": SimpleConvSampler({}),
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
        "resume_from": ["", "Conv", 56, 268, 16, "", "", 2, 3]
    })
    tester.run({})


if __name__ == '__main__':
    layer_latency_test_tpu()
