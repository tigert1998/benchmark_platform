import itertools

import tensorflow as tf

from .utils import shufflenetv2_stages, merge_profiles, op_name_to_model_name
from .samplier import Sampler

import mobilenet.mobilenet_v2 as mobilenet_v2
import mobilenet.conv_blocks as ops


def _get_conv_profiles():
    """Get convolution profiles from ShuffleNetV1, ShuffleNetV2 and MobileNetV2
    Returns:
        [[input_imsize, cin, ksize, cout, stride, [names]]]
    """

    profiles = [
        [224, 3, 3, 24, 2, ["shufflenetv1_conv_0", "shufflenetv2_conv_0"]],
        [7, 704, 1, 1024, 1, ["shufflenetv2_conv_5"]]
    ]

    idx = 1
    for stage in shufflenetv2_stages:
        for block in stage:
            output_imsize, stride, repeat, cout = block
            idx += repeat
            profiles.append([
                output_imsize * min(stride, 2),
                cout // min(stride, 2),
                1, cout // min(stride, 2), 1,
                ["shufflenetv2_block_1stconv_" + str(idx)]
            ])
            profiles.append([
                output_imsize, cout // min(stride, 2),
                1, cout // min(stride, 2), 1,
                ["shufflenetv2_block_2ndconv_" + str(idx)]
            ])

    current_shape = [224, 3]
    for i, op in enumerate(mobilenet_v2.V2_DEF['spec']):
        if op.op == tf.contrib.slim.conv2d:
            profiles.append([
                *current_shape,
                op.params['kernel_size'][0],
                op.params['num_outputs'],
                op.params.get('stride', 1),
                ["mobilenetv2_conv_" + str(i)]])
        elif op.op == ops.expanded_conv:
            expansion_size = op.params.get('expansion_size', mobilenet_v2.V2_DEF['defaults'][(
                ops.expanded_conv,)]['expansion_size'])
            expanded_num_outputs = expansion_size(current_shape[1])
            profiles.append([
                *current_shape,
                1, expanded_num_outputs, 1,
                ["mobilenetv2_bottleneck_1stconv_" + str(i)]])
            profiles.append([
                current_shape[0] // op.params.get("stride",
                                                  1), expanded_num_outputs,
                1, op.params['num_outputs'], 1,
                ["mobilenetv2_bottleneck_2ndconv_" + str(i)]])
        current_shape[0] //= op.params.get("stride", 1)
        current_shape[1] = op.params['num_outputs']

    profiles = merge_profiles(profiles)
    return profiles


class ConvSampler(Sampler):
    @staticmethod
    def get_sample_titles():
        return ["model", "op", "input_imsize", "current_cin", "current_cout",
                "original_cin", "original_cout", "stride", "kernel_size"]

    @staticmethod
    def get_samples():
        for profiles in _get_conv_profiles():
            hash_set = set()
            input_imsize, cin, _, cout, stride, names = profiles
            for op_name, model_name in zip(names, map(op_name_to_model_name, names)):
                if '1st' in op_name:
                    hash_set_key = (model_name, '1st')
                    cin_cout_range = itertools.product([cin], range(
                        int(0.2 * cout), int(2 * cout), 4))
                elif '2nd' in op_name:
                    hash_set_key = (model_name, '2nd')
                    cin_cout_range = itertools.product(
                        range(int(0.2 * cin), int(2 * cin), 4), [cout])
                else:
                    continue
                if hash_set_key in hash_set:
                    continue
                else:
                    hash_set.add(hash_set_key)
                for current_cin, current_cout in cin_cout_range:
                    for ksize in [1, 3, 5, 7]:
                        yield [model_name, "Conv", input_imsize, current_cin, current_cout, cin, cout, stride, ksize]
