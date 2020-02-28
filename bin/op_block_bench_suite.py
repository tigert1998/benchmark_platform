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


def quant_name_from_sdk(inference_sdk):
    quantization = inference_sdk.settings["quantization"]
    if quantization == "":
        return "none"
    else:
        return quantization


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

    # inference_sdks
    inference_sdks = []
    for quantization in ["", "asymmetric_quantized-u8", "dynamic_fixed_point-8", "dynamic_fixed_point-16"]:
        inference_sdks.append(Rknn({
            "rknn_target": None,
            "quantization": quantization,
        }))

    connection = Adb("TD033101190100171", False)

    for tester_class, sampler_class, name in tester_configs:
        for inference_sdk in inference_sdks:
            concrete_tester = tester_class({
                "connection": connection,
                "inference_sdk": inference_sdk,
                "sampler": sampler_class(),
                "dirname": "rknn/{}".format(name),
                "subdir": quant_name_from_sdk(inference_sdk),
                "resume_from": None
            })
            concrete_tester.run({
            })


if __name__ == "__main__":
    rknn_main()
