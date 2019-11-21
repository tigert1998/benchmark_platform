from .sampler import Sampler

from .utils import shufflenetv1_stages, shufflenetv2_stages, merge_profiles, op_name_to_model_name

import mobilenet.mobilenet_v2 as mobilenet_v2
import mobilenet.conv_blocks as ops


def _get_dwconv_profiles():
    """Get depthwise convolution profiles from ShuffleNetV1, ShuffleNetV2 and MobileNetV2
    Returns:
        [[input_imsize, cin, ksize, stride, [names]]]
    """
    profiles = []

    current_shape = [224, 3]
    for i, op in enumerate(mobilenet_v2.V2_DEF['spec']):
        stride = op.params.get("stride", 1)
        if op.op == ops.expanded_conv:
            expansion_size = op.params.get('expansion_size', mobilenet_v2.V2_DEF['defaults'][(
                ops.expanded_conv,)]['expansion_size'])
            expanded_num_outputs = expansion_size(current_shape[1])
            profiles.append([
                current_shape[0], expanded_num_outputs, 3, stride,
                ["mobilenetv2_bottleneck_dwconv_" + str(i)]])
        current_shape[0] //= stride
        current_shape[1] = op.params['num_outputs']

    for net_name, stage_arr in {
        "shufflenetv1": shufflenetv1_stages,
        "shufflenetv2": shufflenetv2_stages
    }.items():
        idx = 1
        for stage in stage_arr:
            for block in stage:
                idx += 1
                output_imsize, stride, _, cout = block
                if stride == 1:
                    if '1' in net_name:
                        profiles.append([
                            output_imsize, cout, 3, 1,
                            [net_name + "_block_dwconv_" + str(idx)]
                        ])
                    else:
                        profiles.append([
                            output_imsize, cout // 2, 3, 1,
                            [net_name + "_block_dwconv_" + str(idx)]
                        ])
                else:
                    profiles.append([
                        output_imsize * stride, cout // stride, 3, stride,
                        [net_name + "_block_dwconv_" + str(idx)]
                    ])

    profiles = merge_profiles(profiles)
    return profiles


class DwconvSampler(Sampler):
    @staticmethod
    def get_sample_titles():
        return ["model", "op", "input_imsize", "current_cin", "current_cout",
                "original_cin", "original_cout", "stride", "kernel_size"]

    def _get_samples_without_filter(self):
        for profiles in _get_dwconv_profiles():
            input_imsize, cin, _, stride, names = profiles
            for model_name in list(set(map(op_name_to_model_name, names))):
                for current_cin in range(int(0.2 * cin), int(2 * cin), 4):
                    for ksize in [3, 5, 7]:
                        yield [model_name, "DWConv", input_imsize, current_cin, current_cin, cin, cin, stride, ksize]
