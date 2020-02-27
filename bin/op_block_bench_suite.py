import itertools

import tensorflow as tf

from testers.inference_sdks.tflite import Tflite

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


def tflite_cpu_main():
    # inference_sdks
    inference_sdks = []
    for quantization in ["", "int", "float16", "weight"]:
        inference_sdks.append(Tflite({
            "benchmark_model_path": "/data/local/tmp/tf-r2.1-60afa4e/benchmark_model",
            "quantization": quantization,
        }))

    connection = Adb("5e6fecf", False)

    for tester_class, sampler_class in [
        (TestConv, OpExperimentConvSampler),
        (TestDwconv, OpExperimentDwconvSampler),
        (TestDilatedConv, DilatedConvSampler),
        (TestGconv, GconvSampler),
        (TestAdd, AddSampler),
        (TestConcat, ConcatSampler),
        (TestGlobalPooling, GlobalPoolingSampler),
        (TestFc, OpExperimentFcSampler),
        (TestShuffle, ShuffleSampler),

        (TestMbnetV1Block, MbnetV1BlockSampler),
        (TestMbnetV2Block, MbnetV2BlockSampler),
        (TestShufflenetV1Unit, ShufflenetV1UnitSampler),
        (TestShufflenetV2Unit, ShufflenetV2UnitSampler),
        (TestResnetV1Block, ResnetV1BlockSampler),
        (TestDenseBlock, DenseBlockSampler),

        (TestMixConv, MixConvSampler)
    ]:
        for inference_sdk in inference_sdks:
            if inference_sdk.settings["quantization"] == "":
                subdir = "none"
            else:
                subdir = inference_sdk.settings["quantization"]

            concrete_tester = tester_class({
                "connection": connection,
                "inference_sdk": inference_sdk,
                "sampler": sampler_class(),
                "subdir": subdir,
                "resume_from": None
            })
            concrete_tester.run({
                "use_gpu": False
            })


if __name__ == "__main__":
    tflite_cpu_main()
