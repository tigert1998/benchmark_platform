from glob import glob
from utils.preprocess import InceptionPreprocess, VggPreprocess
import numpy as np


def accuracy_test_rknn():
    from accuracy_tester.accuracy_tester import AccuracyTester
    from accuracy_tester.data_preparers.data_preparer_def import DataPreparerDef
    from accuracy_tester.accuracy_evaluators.rknn import Rknn

    tester = AccuracyTester({
        "zip_size": 50000,
        "model_paths": glob("C:/Users/v-xiat/Downloads/imagenet/models/*.rknn"),
        "data_preparer": DataPreparerDef({
            "labels_path": "C:/Users/v-xiat/Downloads/imagenet/val_labels.txt",
            "validation_set_path": "C:/Users/v-xiat/Downloads/imagenet/validation",
        }),
        "accuracy_evaluator": Rknn({
            "preprocess": lambda image: InceptionPreprocess.resize(image, 224)
        })
    })
    tester.run()


def accuracy_test_tflite():
    from accuracy_tester.accuracy_tester import AccuracyTester
    from accuracy_tester.data_preparers.android_data_preparer import AndroidDataPreparer
    from accuracy_tester.accuracy_evaluators.tflite import Tflite

    tester = AccuracyTester({
        "dirname": "quick_test",
        "zip_size": 20,
        "model_paths": glob("C:/Users/v-xiat/Microsoft/Shihao Han (FA Talent) - ChannelNas/models/tflite/resnet_v2_50_224/resnet_v2_50_224_int_quant.tflite"),
        "data_preparer": AndroidDataPreparer({
            "labels_path": "C:/Users/v-xiat/Downloads/imagenet/val_labels.txt",
            "validation_set_path": "C:/Users/v-xiat/Downloads/imagenet/validation",
            "adb_device_id": "5e6fecf",
            "skip_dataset_preparation": True,
            "skip_models_preparation": False
        }),
        "accuracy_evaluator": Tflite({
            "eval_on_host": False,
            # on guest
            "adb_device_id": "5e6fecf",
            "imagenet_accuracy_eval_path": "/data/local/tmp/tf-r2.1-60afa4e/imagenet_accuracy_eval",
            "imagenet_accuracy_eval_flags": {
                "delegate": ""
            }
        })
    })
    tester.run()


def model_latency_test():
    from testers.tester_impls.test_model import TestModel
    from testers.inference_sdks.tflite import Tflite
    from testers.sampling.model_sampler import ModelSampler

    tester = TestModel(settings={
        "adb_device_id": "5e6fecf",
        "inference_sdk": Tflite({
            "benchmark_model_path": "/data/local/tmp/tf-r2.1-60afa4e/benchmark_model",
            "su": True
        }),
        "sampler": ModelSampler({
            "model_paths": glob("C:/Users/v-xiat/Microsoft/Shihao Han (FA Talent) - ChannelNas/models/tflite/mobilenet_v2_1.0/mobilenet_v2_1.0_224_frozen.tflite")
        })
    })

    tester.run(benchmark_model_flags={
        "num_runs": 30,
        "use_gpu": True,
        "gpu_precision_loss_allowed": False
    })


def accuracy_test_pb():
    from accuracy_tester.accuracy_tester import AccuracyTester
    from accuracy_tester.data_preparers.data_preparer_def import DataPreparerDef
    from accuracy_tester.accuracy_evaluators.tf_evaluator import TfEvaluator

    tester = AccuracyTester({
        "zip_size": 100,
        "model_paths": ["C:/Users/v-xiat/Downloads/imagenet/models/inception_v4.pb"],
        "data_preparer": DataPreparerDef({
            "labels_path": "C:/Users/v-xiat/Downloads/imagenet/val_labels.txt",
            "validation_set_path": "C:/Users/v-xiat/Downloads/imagenet/validation",
        }),
        "accuracy_evaluator": TfEvaluator({
            "preprocess": lambda image: InceptionPreprocess.preprocess(image, 299),
            "index_to_label": lambda index: str(index)
        })
    })
    tester.run()


if __name__ == '__main__':
    accuracy_test_tflite()
