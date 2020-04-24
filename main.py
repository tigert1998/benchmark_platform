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


def overhead_test():
    from testers.tester_impls.test_stacked import TestStacked
    from testers.sampling.overhead_sampler import OverheadSampler
    from testers.inference_sdks.tpu import Tpu
    from network.resnet_blocks import resnet_v1_block

    tester = TestStacked({
        "inference_sdk": Tpu({
            "edgetpu_compiler_path": "/home/xiaohu/edgetpu/compiler/x86_64/edgetpu_compiler",
            "libedgetpu_path": "/home/xiaohu/edgetpu/libedgetpu/direct/k8/libedgetpu.so.1"
        }),
        "sampler": OverheadSampler(),
        "min": 1,
        "max": 5
    })

    def add_layer(net):
        cin = net.get_shape().as_list()[-1]
        return resnet_v1_block(net, [cin // 4, cin // 4, cin], 1, 3)
    tester.add_layer = add_layer

    tester.run({})


def hardware_computational_intensity():
    from testers.sampling.matmul_sampler import MatmulSampler
    from testers.tester_impls.test_matmul import TestMatmul
    from testers.inference_sdks.tflite_modified import TfliteModified

    tester = TestMatmul({
        "connection": Adb("5e6fecf", True),
        "inference_sdk": TfliteModified({
            "benchmark_model_path": "/data/local/tmp/tf-r2.1-60afa4e/benchmark_model_modified",
        }),
        "sampler": MatmulSampler({}),
    })
    tester.run({
        "use_gpu": True,
        "work_group_size": "",
        "tuning_type": "EXHAUSTIVE",
        "kernel_path": "/data/local/tmp/kernel.cl"
    })


def model_latency_test():
    from testers.tester_impls.test_model import TestModel
    from testers.inference_sdks.tflite import Tflite
    # from testers.inference_sdks.tpu import Tpu
    # from testers.inference_sdks.rknn import Rknn
    from testers.sampling.model_sampler import ModelSampler

    tester = TestModel(settings={
        "connection": Adb("5e6fecf", True),
        "inference_sdk": Tflite({
            "benchmark_model_path": "/data/local/tmp/tf-r2.1-60afa4e/benchmark_model",
        }),
        "sampler": ModelSampler({
            "model_details":
                get_model_details(
                    ["resnet_v1"], "tflite",
                    ["", "float16"], "cpu"
                )
        })
    })

    tester.run({
        "use_gpu": True
        # "use_gpu": True,
        # "kernel_path": "/data/local/tmp/kernel.cl",
        # "precision": "F32"
    })


def model_flops_test():
    from testers.tester_impls.test_model import TestModel
    from testers.inference_sdks.flops_calculator import FlopsCalculator
    from testers.sampling.model_sampler import ModelSampler

    tester = TestModel(settings={
        "connection": Connection(),
        "inference_sdk": FlopsCalculator({}),
        "sampler": ModelSampler({
            "model_details": get_model_details(["resnet_v1"], "pb", ["patched"])
        })
    })

    tester.run(benchmark_model_flags={})


def accuracy_test_rknn():
    from accuracy_tester.accuracy_tester import AccuracyTester
    from accuracy_tester.accuracy_evaluators.rknn import Rknn

    tester = AccuracyTester({
        "dirname": "test_rknn",
        "zip_size": 50000,
        "dataset_size": 100,
        "model_details": get_model_details(["proxyless"], "rknn", [
            # "", "dynamic_fixed_point_16",
            "dynamic_fixed_point_8", "asymmetric_quantized_u8"
        ]),
        "data_preparer": DataPreparerDef({
            "labels_path": "C:/Users/v-xiat/Downloads/playground/imagenet/val.txt",
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
        "model_details": get_model_details(["mnasnet"], "pb", ["patched"]),
        "data_preparer": DataPreparerDef({
            "labels_path": "C:/Users/tigertang/Projects/dataset/val_labels.txt",
            "validation_set_path": "C:/Users/tigertang/Projects/dataset/validation",
            "skip_dataset_preparation": True,
            "skip_models_preparation": True,
        }),
        "accuracy_evaluator": TfEvaluator({})
    })
    tester.run()


def accuracy_test_onnx():
    from accuracy_tester.accuracy_tester import AccuracyTester
    from accuracy_tester.accuracy_evaluators.onnx import Onnx

    tester = AccuracyTester({
        "zip_size": 50000,
        "dataset_size": 100,
        "model_details": get_model_details(["shufflenet_v1"], "onnx", [""]),
        "data_preparer": DataPreparerDef({
            "labels_path": "C:/Users/tigertang/Projects/dataset/val_labels.txt",
            "validation_set_path": "C:/Users/tigertang/Projects/dataset/validation",
            "skip_dataset_preparation": True,
            "skip_models_preparation": True,
        }),
        "accuracy_evaluator": Onnx({})
    })
    tester.run()


def accuracy_test_tpu():
    from accuracy_tester.accuracy_tester import AccuracyTester
    from accuracy_tester.accuracy_evaluators.tpu import Tpu

    tester = AccuracyTester({
        "zip_size": 50000,
        "dataset_size": 50000,
        "model_details": get_model_details(None, "tflite", ["edgetpu"], "edgetpu"),
        "data_preparer": DataPreparerDef({
            "labels_path": "/home/xiaohu/val_labels.txt",
            "validation_set_path": "/home/hanxiao/benchmarks/imagenet_dataset",
            "skip_dataset_preparation": True,
            "skip_models_preparation": True,
        }),
        "accuracy_evaluator": Tpu({})
    })
    tester.run()


def accuracy_test_tflite():
    from accuracy_tester.accuracy_tester import AccuracyTester
    from accuracy_tester.accuracy_evaluators.tflite import Tflite

    tester = AccuracyTester({
        "zip_size": 50000,
        "dataset_size": 100,
        "model_details": get_model_details(["resnet_v1"], "tflite", ["int"], "cpu"),
        "data_preparer": AndroidDataPreparer({
            "connection": Adb("5e6fecf", False),
            "labels_path": "C:/Users/v-xiat/Downloads/playground/imagenet/val.txt",
            "validation_set_path": "C:/Users/v-xiat/Downloads/playground/imagenet/validation",
            "skip_dataset_preparation": True,
            "skip_models_preparation": False,
        }),
        "accuracy_evaluator": Tflite({
            "connection": Adb("5e6fecf", False),
            "imagenet_accuracy_eval_path": "/data/local/tmp/tf-r2.1-60afa4e/imagenet_accuracy_eval",
            "imagenet_accuracy_eval_flags": {
                "delegate": "gpu",
            },
            "charging_opts": {
                "min": 0.8, "max": 0.95
            }
        })
    })
    tester.run()


def layer_latency_test_tflite():
    from testers.inference_sdks.tflite_modified import TfliteModified
    from testers.inference_sdks.tflite import Tflite

    def conv_sampler_filter(sample):
        _, _, input_imsize, cin, cout, _, _, stride, ksize = sample
        return input_imsize == 28 and cin == 320 and ksize == 3 and\
            stride == 1 and 200 <= cout and cout <= 400

    def dwconv_sampler_filter(sample):
        _, _, input_imsize, cin, _, _, _, stride, ksize = sample
        return input_imsize == 7 and ksize == 3 and stride == 1 and\
            cin in range(200, 401)

    tester = TestDwconv({
        "connection": Adb("5e6fecf", True),
        "inference_sdk": Tflite({
            "benchmark_model_path": "/data/local/tmp/tf-r2.1-60afa4e/benchmark_model",
        }),
        "sampler": ChannelExperimentDwconvSampler({
            "filter": dwconv_sampler_filter,
            "channel_step": 1
        }),
    })
    tester.run({
        "use_gpu": False,
        # "use_gpu": True,
        # "work_group_size": "",
        # "tuning_type": "EXHAUSTIVE",
        # "kernel_path": "/data/local/tmp/kernel.cl"
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

    def conv_sampler_filter(sample):
        _, _, input_imsize, cin, cout, _, _, stride, ksize = sample
        return input_imsize == 28 and cin == 320 and ksize == 3 and\
            stride == 1 and (cout in range(221, 241))

    tester = TestConv({
        "connection": Adb("TD033101190100171", False),
        "inference_sdk": Rknn({
            "rknn_target": None,
            "quantization": "asymmetric_quantized-u8"
        }),
        "sampler": ChannelExperimentConvSampler({
            "channel_step": 1,
            "filter": conv_sampler_filter
        }),
        "resume_from": None
    })
    tester.run({})


if __name__ == '__main__':
    accuracy_test_onnx()
