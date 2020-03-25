# ops sampler
from testers.sampling.conv_sampler import OpExperimentConvSampler
from testers.sampling.dwconv_sampler import OpExperimentDwconvSampler
from testers.sampling.dilated_conv_sampler import DilatedConvSampler
from testers.sampling.gconv_sampler import GconvSampler
from testers.sampling.elementwise_ops_sampler import \
    AddSampler, ConcatSampler, GlobalPoolingSampler, ShuffleSampler
from testers.sampling.fc_sampler import OpExperimentFcSampler

# blocks sampler
from testers.sampling.mbnet_blocks_sampler import \
    MbnetV1BlockSampler, MbnetV2BlockSampler
from testers.sampling.shufflenet_units_sampler import \
    ShufflenetV1UnitSampler, ShufflenetV2UnitSampler
from testers.sampling.resnet_blocks_sampler import ResnetV1BlockSampler
from testers.sampling.dense_blocks_sampler import DenseBlockSampler

# new ops sampler
from testers.sampling.mix_conv_sampler import MixConvSampler

# ops testers
from testers.tester_impls.test_conv import TestConv
from testers.tester_impls.test_dwconv import TestDwconv
from testers.tester_impls.test_dilated_conv import TestDilatedConv
from testers.tester_impls.test_gconv import TestGconv
from testers.tester_impls.test_elementwise_ops import \
    TestAdd, TestConcat, TestGlobalPooling, TestShuffle
from testers.tester_impls.test_fc import TestFc

# block testers
from testers.tester_impls.test_mbnet_blocks import \
    TestMbnetV1Block, TestMbnetV2Block
from testers.tester_impls.test_shufflenet_units import \
    TestShufflenetV1Unit, TestShufflenetV2Unit
from testers.tester_impls.test_resnet_v1_block import TestResnetV1Block
from testers.tester_impls.test_dense_block import TestDenseBlock

# new ops testers
from testers.tester_impls.test_mix_conv import TestMixConv

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

    tester_configs = [
        (TestConv, OpExperimentConvSampler(), "conv", {}),
        (TestDwconv, OpExperimentDwconvSampler(), "dwconv", {}),
        (TestDilatedConv, DilatedConvSampler(), "dilated_conv", {}),
        # (TestGconv, GconvSampler(), "gconv", {}),
        (TestAdd, AddSampler(), "add", {"num_write_kernels": 2}),
        (TestConcat, ConcatSampler(), "concat", {"num_write_kernels": 2}),
        (TestGlobalPooling, GlobalPoolingSampler(), "global_pooling", {}),
        (TestFc, OpExperimentFcSampler(), "fc", {}),
        (TestShuffle, ShuffleSampler(), "shuffle", {}),

        (TestMbnetV1Block, MbnetV1BlockSampler(), "mbnet_v1_block", {}),
        (TestMbnetV2Block, MbnetV2BlockSampler(), "mbnet_v2_block", {}),
        # (TestShufflenetV1Unit, ShufflenetV1UnitSampler(), "shufflenet_v1_unit", {}),
        (TestShufflenetV2Unit, ShufflenetV2UnitSampler({
            "filter": lambda sample: sample[-2] == 2
        }), "shufflenet_v2_unit", {}),
        (TestResnetV1Block, ResnetV1BlockSampler(), "resnet_v1_block", {}),
        (TestDenseBlock, DenseBlockSampler(), "dense_block", {}),

        # (TestMixConv, MixConvSampler(), "mix_conv", {})
    ]

    # inference_sdks
    inference_sdks = []
    for quantization in ["", "float16"]:
        inference_sdks.append(TfliteModified({
            "benchmark_model_path": "/data/local/tmp/tf-r2.1-60afa4e/benchmark_model_modified",
            "quantization": quantization,
        }))

    connection = Adb("5e6fecf", True)

    for tester_class, sampler, name, additional_flags in tester_configs:
        for inference_sdk in inference_sdks:
            concrete_tester = tester_class({
                "connection": connection,
                "inference_sdk": inference_sdk,
                "sampler": sampler,
                "dirname": "gpu/{}".format(name),
                "subdir": quant_name_from_sdk(inference_sdk),
                "resume_from": None
            })
            concrete_tester.run({
                "use_gpu": True,
                "work_group_size": "",
                "tuning_type": "EXHAUSTIVE",
                "kernel_path": "/data/local/tmp/kernel.cl",
                **additional_flags
            })


def tflite_cpu_main():
    from testers.inference_sdks.tflite import Tflite

    tester_configs = [
        (TestConv, OpExperimentConvSampler, "conv"),
        (TestDwconv, OpExperimentDwconvSampler, "dwconv"),
        (TestDilatedConv, DilatedConvSampler, "dilated_conv"),
        (TestGconv, GconvSampler, "gconv"),
        (TestAdd, AddSampler, "add"),
        (TestConcat, ConcatSampler, "concat"),
        (TestGlobalPooling, GlobalPoolingSampler, "global_pooling"),
        (TestFc, OpExperimentFcSampler, "fc"),
        (TestShuffle, ShuffleSampler, "shuffle"),

        (TestMbnetV1Block, MbnetV1BlockSampler, "mbnet_v1_block"),
        (TestMbnetV2Block, MbnetV2BlockSampler, "mbnet_v2_block"),
        (TestShufflenetV1Unit, ShufflenetV1UnitSampler, "shufflenet_v1_unit"),
        (TestShufflenetV2Unit, ShufflenetV2UnitSampler, "shufflenet_v2_unit"),
        (TestResnetV1Block, ResnetV1BlockSampler, "resnet_v1_block"),
        (TestDenseBlock, DenseBlockSampler, "dense_block"),

        (TestMixConv, MixConvSampler, "mix_conv")
    ]

    # inference_sdks
    inference_sdks = []
    for quantization in ["", "int", "float16", "weight"]:
        inference_sdks.append(Tflite({
            "benchmark_model_path": "/data/local/tmp/tf-r2.1-60afa4e/benchmark_model",
            "quantization": quantization,
        }))

    connection = Adb("5e6fecf", False)

    for tester_class, sampler_class, name in tester_configs:
        for inference_sdk in inference_sdks:
            concrete_tester = tester_class({
                "connection": connection,
                "inference_sdk": inference_sdk,
                "sampler": sampler_class(),
                "dirname": "cpu/{}".format(name),
                "subdir": quant_name_from_sdk(inference_sdk),
                "resume_from": None
            })
            concrete_tester.run({
                "use_gpu": False
            })


def rknn_main():
    from testers.inference_sdks.rknn import Rknn

    def always_true(quant_name: str, sample):
        return True

    def mbnet_v2_block_sampler_filter(quant_name: str, sample):
        _, input_imsize, cin, cout, with_se, stride, ksize = sample
        if with_se:
            return False
        if quant_name == "none" or quant_name == "dynamic_fixed_point-16":
            if [input_imsize, cin, cout] >= [7, 240, 480]:
                return ksize == 3
            return True
        else:
            return True

    def shufflenet_v2_unit_sampler_filter(quant_name: str, sample):
        _, imsize, cin, stride, ksize = sample
        return stride == 2

    tester_configs = [
        (TestConv, OpExperimentConvSampler, "conv", always_true),
        (TestDwconv, OpExperimentDwconvSampler, "dwconv", always_true),
        (TestDilatedConv, DilatedConvSampler, "dilated_conv", always_true),
        # (TestGconv, GconvSampler, "gconv", always_true),
        (TestAdd, AddSampler, "add", always_true),
        # (TestConcat, ConcatSampler, "concat", always_true),
        (TestGlobalPooling, GlobalPoolingSampler, "global_pooling", always_true),
        (TestFc, OpExperimentFcSampler, "fc", always_true),
        (TestShuffle, ShuffleSampler, "shuffle", always_true),

        (TestMbnetV1Block, MbnetV1BlockSampler, "mbnet_v1_block", always_true),
        (TestMbnetV2Block, MbnetV2BlockSampler,
         "mbnet_v2_block", mbnet_v2_block_sampler_filter),
        # (TestShufflenetV1Unit, ShufflenetV1UnitSampler, "shufflenet_v1_unit", always_true),
        (TestShufflenetV2Unit, ShufflenetV2UnitSampler,
         "shufflenet_v2_unit", shufflenet_v2_unit_sampler_filter),
        (TestResnetV1Block, ResnetV1BlockSampler, "resnet_v1_block", always_true),
        (TestDenseBlock, DenseBlockSampler, "dense_block", always_true),

        # (TestMixConv, MixConvSampler, "mix_conv", always_true)
    ]

    # inference_sdks
    inference_sdks = []
    for quantization in ["", "asymmetric_quantized-u8", "dynamic_fixed_point-8", "dynamic_fixed_point-16"]:
        inference_sdks.append(Rknn({
            "rknn_target": None,
            "quantization": quantization,
        }))

    connection = Adb("TD033101190100171", False)

    for tester_class, sampler_class, name, sampler_filter in tester_configs:
        for inference_sdk in inference_sdks:
            quant_name = quant_name_from_sdk(inference_sdk)
            concrete_tester = tester_class({
                "connection": connection,
                "inference_sdk": inference_sdk,
                "sampler": sampler_class({"filter": lambda sample: sampler_filter(quant_name, sample)}),
                "dirname": "rknn/{}".format(name),
                "subdir": quant_name,
                "resume_from": None
            })
            concrete_tester.run({
                "disable_timeout": True
            })


def tflite_tpu_main():
    from testers.inference_sdks.tpu import Tpu

    tester_configs = [
        (TestConv, OpExperimentConvSampler, "conv"),
        (TestDwconv, OpExperimentDwconvSampler, "dwconv"),
        (TestDilatedConv, DilatedConvSampler, "dilated_conv"),
        (TestGconv, GconvSampler, "gconv"),
        (TestAdd, AddSampler, "add"),
        (TestConcat, ConcatSampler, "concat"),
        (TestGlobalPooling, GlobalPoolingSampler, "global_pooling"),
        (TestFc, OpExperimentFcSampler, "fc"),
        # (TestShuffle, ShuffleSampler, "shuffle"),

        (TestMbnetV1Block, MbnetV1BlockSampler, "mbnet_v1_block"),
        (TestMbnetV2Block, MbnetV2BlockSampler, "mbnet_v2_block"),
        # (TestShufflenetV1Unit, ShufflenetV1UnitSampler, "shufflenet_v1_unit"),
        # (TestShufflenetV2Unit, ShufflenetV2UnitSampler, "shufflenet_v2_unit"),
        (TestResnetV1Block, ResnetV1BlockSampler, "resnet_v1_block"),
        (TestDenseBlock, DenseBlockSampler, "dense_block"),

        (TestMixConv, MixConvSampler, "mix_conv")
    ]

    # inference_sdks
    inference_sdks = [
        Tpu({
            "edgetpu_compiler_path": "/home/xiaohu/edgetpu/compiler/x86_64/edgetpu_compiler",
            "libedgetpu_path": "/home/xiaohu/edgetpu/libedgetpu/direct/k8/libedgetpu.so.1"
        })
    ]

    connection = Connection()

    for tester_class, sampler_class, name in tester_configs:
        for inference_sdk in inference_sdks:
            concrete_tester = tester_class({
                "connection": connection,
                "inference_sdk": inference_sdk,
                "sampler": sampler_class(),
                "dirname": "tpu/{}".format(name),
                "resume_from": None
            })
            concrete_tester.run({})


def flops_main():
    from testers.inference_sdks.flops_calculator import FlopsCalculator

    tester_configs = [
        (TestConv, OpExperimentConvSampler, "conv"),
        (TestDwconv, OpExperimentDwconvSampler, "dwconv"),
        (TestDilatedConv, DilatedConvSampler, "dilated_conv"),
        (TestGconv, GconvSampler, "gconv"),
        (TestAdd, AddSampler, "add"),
        (TestConcat, ConcatSampler, "concat"),
        (TestGlobalPooling, GlobalPoolingSampler, "global_pooling"),
        (TestFc, OpExperimentFcSampler, "fc"),
        (TestShuffle, ShuffleSampler, "shuffle"),

        (TestMbnetV1Block, MbnetV1BlockSampler, "mbnet_v1_block"),
        (TestMbnetV2Block, MbnetV2BlockSampler, "mbnet_v2_block"),
        (TestShufflenetV1Unit, ShufflenetV1UnitSampler, "shufflenet_v1_unit"),
        (TestShufflenetV2Unit, ShufflenetV2UnitSampler, "shufflenet_v2_unit"),
        (TestResnetV1Block, ResnetV1BlockSampler, "resnet_v1_block"),
        (TestDenseBlock, DenseBlockSampler, "dense_block"),

        (TestMixConv, MixConvSampler, "mix_conv")
    ]

    # inference_sdks
    inference_sdks = [FlopsCalculator()]

    connection = Connection()

    for tester_class, sampler_class, name in tester_configs:
        for inference_sdk in inference_sdks:
            concrete_tester = tester_class({
                "connection": connection,
                "inference_sdk": inference_sdk,
                "sampler": sampler_class(),
                "dirname": "flops/{}".format(name),
                "resume_from": None
            })
            concrete_tester.run({})


if __name__ == "__main__":
    tflite_gpu_main()
