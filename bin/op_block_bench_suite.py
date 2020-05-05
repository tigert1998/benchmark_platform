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


def always_true(*args, **kwargs):
    return True


def tflite_gpu_main():
    from testers.inference_sdks.tflite_modified import TfliteModified

    def shufflenet_v2_unit_sampler_filter(sample):
        return sample[-2] == 2

    tester_configs = [
        (TestConv, OpExperimentConvSampler, "conv", always_true),
        (TestDwconv, OpExperimentDwconvSampler, "dwconv", always_true),
        (TestDilatedConv, DilatedConvSampler, "dilated_conv", always_true),
        # (TestGconv, GconvSampler, "gconv",always_true),
        (TestAdd, AddSampler, "add", always_true),
        (TestConcat, ConcatSampler, "concat", always_true),
        (TestGlobalPooling, GlobalPoolingSampler, "global_pooling", always_true),
        (TestFc, OpExperimentFcSampler, "fc", always_true),
        (TestShuffle, ShuffleSampler, "shuffle", always_true),

        (TestMbnetV1Block, MbnetV1BlockSampler, "mbnet_v1_block", always_true),
        (TestMbnetV2Block, MbnetV2BlockSampler, "mbnet_v2_block", always_true),
        # (TestShufflenetV1Unit, ShufflenetV1UnitSampler, "shufflenet_v1_unit", always_true),
        (TestShufflenetV2Unit, ShufflenetV2UnitSampler,
         "shufflenet_v2_unit", shufflenet_v2_unit_sampler_filter),
        (TestResnetV1Block, ResnetV1BlockSampler, "resnet_v1_block", always_true),
        (TestDenseBlock, DenseBlockSampler, "dense_block", always_true),

        # (TestMixConv, MixConvSampler, "mix_conv", always_true),
    ]

    # inference_sdks
    inference_sdks = []
    for quantization in ["", "float16"]:
        inference_sdks.append(TfliteModified({
            "benchmark_model_path": "/data/local/tmp/tf-r2.1-60afa4e/benchmark_model_modified",
            "quantization": quantization,
        }))

    connection = Adb("5e6fecf", True)

    for tester_class, sampler_class, name, sampler_filter in tester_configs:
        for inference_sdk in inference_sdks:
            concrete_tester = tester_class({
                "connection": connection,
                "inference_sdk": inference_sdk,
                "sampler": sampler_class({"filter": sampler_filter}),
                "dirname": "gpu/{}".format(name),
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

        (TestMixConv, MixConvSampler, "mix_conv"),
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

    def gconv_sampler_filter(quant_name: str, sample):
        _, input_imsize, cin, cout, num_groups, stride, ksize = sample
        return num_groups not in [3, 8]

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

    def shufflenet_v1_unit_sampler_filter(quant_name: str, sample):
        _, input_imsize, cin, cout, num_groups, mid_channels, stride, ksize = sample
        return num_groups not in [3, 8]

    def mix_conv_sampler_filter(quant_name: str, sample):
        _, input_imsize, cin, cout, num_groups, stride = sample
        return num_groups not in [3, 5]

    def gen_pad_func(name: str):
        import tensorflow as tf

        def pad_before_input(shapes):
            assert len(shapes) == 1
            shape = shapes[0]
            assert len(shape) == 4 and shape[1] == shape[2] and shape[0] == 1
            input_tensor = tf.placeholder(
                name="input_im_0",
                dtype=tf.float32,
                shape=shape
            )
            net = tf.keras.layers.Conv2D(
                filters=shape[-1],
                kernel_size=[3, 3],
                strides=[1, 1],
                padding='same'
            )(input_tensor)

            return [input_tensor], [net]

        def pad_after_output(output_tensors):
            assert len(output_tensors) == 1
            net = output_tensors[0]
            cout = net.get_shape().as_list()[-1]
            net = tf.keras.layers.Conv2D(
                filters=cout,
                kernel_size=[3, 3],
                strides=[1, 1],
                padding='same'
            )(net)
            return [net]

        if name in ["add", "concat", "global_pooling", "fc"]:
            return None, None
        elif name in [
            "conv", "dwconv", "dilated_conv", "gconv",
            "shuffle",
            "mbnet_v1_block", "mbnet_v2_block",
            "shufflenet_v1_unit", "shufflenet_v2_unit",
            "resnet_v1_block", "dense_block",
            "mix_conv"
        ]:
            return pad_before_input, pad_after_output
        else:
            assert False

    tester_configs = [
        (TestConv, OpExperimentConvSampler, "conv", always_true),
        (TestDwconv, OpExperimentDwconvSampler, "dwconv", always_true),
        (TestDilatedConv, DilatedConvSampler, "dilated_conv", always_true),
        (TestGconv, GconvSampler, "gconv", gconv_sampler_filter),
        (TestAdd, AddSampler, "add", always_true),
        (TestConcat, ConcatSampler, "concat", always_true),
        (TestGlobalPooling, GlobalPoolingSampler, "global_pooling", always_true),
        (TestFc, OpExperimentFcSampler, "fc", always_true),
        (TestShuffle, ShuffleSampler, "shuffle", always_true),

        (TestMbnetV1Block, MbnetV1BlockSampler, "mbnet_v1_block", always_true),
        (TestMbnetV2Block, MbnetV2BlockSampler,
         "mbnet_v2_block", mbnet_v2_block_sampler_filter),
        (TestShufflenetV1Unit, ShufflenetV1UnitSampler,
         "shufflenet_v1_unit", shufflenet_v1_unit_sampler_filter),
        (TestShufflenetV2Unit, ShufflenetV2UnitSampler,
         "shufflenet_v2_unit", always_true),
        (TestResnetV1Block, ResnetV1BlockSampler, "resnet_v1_block", always_true),
        (TestDenseBlock, DenseBlockSampler, "dense_block", always_true),

        (TestMixConv, MixConvSampler, "mix_conv", mix_conv_sampler_filter),
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

            pad_before_input, pad_after_output = gen_pad_func(name)
            if pad_before_input is not None:
                concrete_tester._pad_before_input = pad_before_input
            if pad_after_output is not None:
                concrete_tester._pad_after_output = pad_after_output

            concrete_tester.run({
                "disable_timeout": True
            })


def tflite_tpu_main():
    from testers.inference_sdks.tpu import Tpu

    def mbnet_v2_block_sampler_filter(sample):
        _, input_imsize, cin, cout, with_se, stride, ksize = sample
        if with_se:
            dic = {
                7: [(7, 320), (3, 512)],
                14: [(7, 240), (5, 320)],
                56: [(3, 64)],
                112: [(3, 32)],
                224: [(3, 32)]
            }
        else:
            dic = {
                14: [(7, 320)]
            }
        if input_imsize not in dic:
            return True
        else:
            for limit_ksize, limit_cin in dic[input_imsize]:
                if ksize >= limit_ksize:
                    return cin < limit_cin
            return True

    def gconv_sampler_filter(sample):
        _, input_imsize, cin, cout, num_groups, stride, ksize = sample
        return input_imsize < 224

    def dense_blocks_sampler_filter(sample):
        _, input_imsize, cin, growth_rate, num_layers, ksize = sample
        dic = {
            28: [(3, 128)],
            112: [(7, 32), (3, 96)],
            224: [(3, 32)]
        }
        if input_imsize not in dic:
            return True
        else:
            for limit_ksize, limit_cin in dic[input_imsize]:
                if ksize >= limit_ksize:
                    return cin < limit_cin
            return True

    def mix_conv_sampler_filter(sample):
        _, input_imsize, cin, cout, num_groups, stride = sample
        if input_imsize >= 224:
            return False
        if input_imsize == 112:
            if num_groups == 4:
                return cin < 96
        return True

    tester_configs = [
        (TestConv, OpExperimentConvSampler, "conv", always_true),
        (TestDwconv, OpExperimentDwconvSampler, "dwconv", always_true),
        (TestDilatedConv, DilatedConvSampler, "dilated_conv", always_true),
        (TestGconv, GconvSampler, "gconv", gconv_sampler_filter),
        (TestAdd, AddSampler, "add", always_true),
        (TestConcat, ConcatSampler, "concat", always_true),
        (TestGlobalPooling, GlobalPoolingSampler, "global_pooling", always_true),
        (TestFc, OpExperimentFcSampler, "fc", always_true),
        # (TestShuffle, ShuffleSampler, "shuffle",always_true),

        (TestMbnetV1Block, MbnetV1BlockSampler, "mbnet_v1_block", always_true),
        (TestMbnetV2Block, MbnetV2BlockSampler,
         "mbnet_v2_block", mbnet_v2_block_sampler_filter),
        # (TestShufflenetV1Unit, ShufflenetV1UnitSampler, "shufflenet_v1_unit",always_true),
        # (TestShufflenetV2Unit, ShufflenetV2UnitSampler, "shufflenet_v2_unit",always_true),
        (TestResnetV1Block, ResnetV1BlockSampler, "resnet_v1_block", always_true),
        (TestDenseBlock, DenseBlockSampler,
         "dense_block", dense_blocks_sampler_filter),

        (TestMixConv, MixConvSampler, "mix_conv", mix_conv_sampler_filter),
    ]

    # inference_sdks
    inference_sdks = [
        Tpu({
            "edgetpu_compiler_path": "/home/xiaohu/edgetpu/compiler/x86_64/edgetpu_compiler",
            "libedgetpu_path": "/home/xiaohu/edgetpu/libedgetpu/direct/k8/libedgetpu.so.1"
        })
    ]

    connection = Connection()

    for tester_class, sampler_class, name, sampler_filter in tester_configs:
        for inference_sdk in inference_sdks:
            concrete_tester = tester_class({
                "connection": connection,
                "inference_sdk": inference_sdk,
                "sampler": sampler_class({"filter": sampler_filter}),
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

        (TestMixConv, MixConvSampler, "mix_conv"),
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
    rknn_main()
