from testers.tester_impls.test_elementwise_ops import TestActivation
from testers.sampling.elementwise_ops_sampler import ActivationSampler

from utils.connection import Adb
from utils.connection import Connection


def quant_name_from_sdk(inference_sdk):
    quantization = inference_sdk.settings["quantization"]
    if quantization == "":
        return "none"
    else:
        return quantization


def tflite_gpu_main():
    from testers.inference_sdks.tflite_modified import TfliteModified

    # inference_sdks
    inference_sdks = []
    for quantization in ["", "float16"]:
        inference_sdks.append(TfliteModified({
            "benchmark_model_path": "/data/local/tmp/tf-r2.1-60afa4e/benchmark_model_modified",
            "quantization": quantization,
        }))

    connection = Adb("5e6fecf", True)

    for inference_sdk in inference_sdks:
        concrete_tester = TestActivation({
            "enable_single_test": True,

            "connection": connection,
            "inference_sdk": inference_sdk,
            "sampler": ActivationSampler(),
            "dirname": "gpu/activation",
            "subdir": quant_name_from_sdk(inference_sdk),
            "resume_from": None
        })
        concrete_tester.run({
            "use_gpu": True,
            "work_group_size": "",
            "tuning_type": "EXHAUSTIVE",
            "kernel_path": "/data/local/tmp/kernel.cl",
        })


def tflite_cpu_main():
    from testers.inference_sdks.tflite import Tflite

    # inference_sdks
    inference_sdks = []
    for quantization in ["", "int", "float16", "weight"]:
        inference_sdks.append(Tflite({
            "benchmark_model_path": "/data/local/tmp/tf-r2.1-60afa4e/benchmark_model",
            "quantization": quantization,
        }))

    connection = Adb("5e6fecf", False)

    for inference_sdk in inference_sdks:
        concrete_tester = TestActivation({
            "enable_single_test": True,

            "connection": connection,
            "inference_sdk": inference_sdk,
            "sampler": ActivationSampler(),
            "dirname": "cpu/activation",
            "subdir": quant_name_from_sdk(inference_sdk),
            "resume_from": None
        })
        concrete_tester.run({
            "use_gpu": False
        })


def tflite_tpu_main():
    from testers.inference_sdks.tpu import Tpu

    def activation_sampler_filter(sample):
        op, input_imsize, cin = sample
        return not (op in ["hardswish"])

    # inference_sdks
    inference_sdks = [
        Tpu({
            "edgetpu_compiler_path": "/home/xiaohu/edgetpu/compiler/x86_64/edgetpu_compiler",
            "libedgetpu_path": "/home/xiaohu/edgetpu/libedgetpu/direct/k8/libedgetpu.so.1"
        })
    ]

    connection = Connection()

    for inference_sdk in inference_sdks:
        concrete_tester = TestActivation({
            "enable_single_test": True,

            "connection": connection,
            "inference_sdk": inference_sdk,
            "sampler": ActivationSampler({"filter": activation_sampler_filter}),
            "dirname": "tpu/activation",
            "resume_from": None
        })
        concrete_tester.run({})


def rknn_main():
    from testers.inference_sdks.rknn import Rknn

    # inference_sdks
    inference_sdks = []
    for quantization in ["", "asymmetric_quantized-u8", "dynamic_fixed_point-8", "dynamic_fixed_point-16"]:
        inference_sdks.append(Rknn({
            "rknn_target": None,
            "quantization": quantization,
        }))

    connection = Adb("TD033101190100171", False)

    def activation_sampler_filter(quant_name: str, sample):
        op, imsize, cin = sample
        return op not in ["swish"]

    for inference_sdk in inference_sdks:
        quant_name = quant_name_from_sdk(inference_sdk)
        concrete_tester = TestActivation({
            "enable_single_test": True,

            "connection": connection,
            "inference_sdk": inference_sdk,
            "sampler": ActivationSampler({
                "filter": lambda sample: activation_sampler_filter(quant_name, sample)
            }),
            "dirname": "rknn/activation",
            "subdir": quant_name,
            "resume_from": None
        })
        concrete_tester.run({
            "disable_timeout": True
        })


if __name__ == "__main__":
    tflite_tpu_main()
