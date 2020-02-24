import itertools

import tensorflow as tf

from .utils import \
    shufflenetv2_stages, merge_profiles, op_name_to_model_name, align, \
    sparse_channels_from_imsize, available_imsizes  # for op experiments
from .sampler import Sampler


def _get_conv_profiles():
    """Get convolution profiles from ShuffleNetV1, ShuffleNetV2 and MobileNetV2
    Returns:
        [[input_imsize, cin, ksize, cout, stride, [names]]]
    """

    import mobilenet.mobilenet_v2 as mobilenet_v2
    import mobilenet.conv_blocks as ops

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
    def default_settings():
        return {
            **Sampler.default_settings(),
            "channel_range": (0.2, 2),
            "channel_step": 4,
        }

    @staticmethod
    def get_sample_titles():
        return ["model", "op", "input_imsize", "current_cin", "current_cout",
                "original_cin", "original_cout", "stride", "kernel_size"]

    def _get_samples_without_filter(self):
        channel_step = self.settings["channel_step"]
        channel_range = self.settings["channel_range"]

        for profiles in _get_conv_profiles():
            hash_set = set()
            input_imsize, cin, _, cout, stride, names = profiles
            for op_name, model_name in zip(names, map(op_name_to_model_name, names)):
                if '1st' in op_name:
                    hash_set_key = (model_name, '1st')
                    cin_cout_range = itertools.product(
                        [cin], range(
                            align(int(channel_range[0] * cout), 2),
                            align(int(channel_range[1] * cout), 2),
                            channel_step
                        )
                    )
                elif '2nd' in op_name:
                    hash_set_key = (model_name, '2nd')
                    cin_cout_range = itertools.product(
                        range(
                            align(int(channel_range[0] * cin), 2),
                            align(int(channel_range[1] * cin), 2),
                            channel_step
                        ), [cout])
                else:
                    continue
                if hash_set_key in hash_set:
                    continue
                else:
                    hash_set.add(hash_set_key)
                for current_cin, current_cout in cin_cout_range:
                    for ksize in [1, 3, 5, 7]:
                        sample = [model_name, "Conv", input_imsize,
                                  current_cin, current_cout, cin, cout, stride, ksize]
                        if self.settings["filter"](sample):
                            yield sample


class ChannelExperimentConvSampler(ConvSampler):
    @staticmethod
    def default_settings():
        return {
            **Sampler.default_settings(),
        }

    def _get_channel_step(self, input_imsize):
        if input_imsize <= 56:
            return 4
        elif input_imsize <= 224:
            return 2
        else:
            assert False

    def _get_channel_range(self, input_imsize):
        channel_step = self._get_channel_step(input_imsize)
        if input_imsize <= 56:
            return list(range(16, 1000 + channel_step, channel_step))
        elif input_imsize <= 112:
            return list(range(12, 320 + channel_step, channel_step))
        elif input_imsize <= 224:
            return [3] + list(range(4, 64 + channel_step, channel_step))
        else:
            assert False

    def _get_cin_cout_range_with_fixed_cin(self, input_imsize):
        cout_range = self._get_channel_range(input_imsize)

        if input_imsize <= 28:
            cin_range = [64, 160, 320, 640]
        elif input_imsize <= 56:
            cin_range = [16, 32, 64, 160]
        elif input_imsize <= 112:
            cin_range = [16, 32, 64]
        elif input_imsize <= 224:
            cin_range = [3]
        else:
            assert False

        return itertools.product(cin_range, cout_range)

    def _get_cin_cout_range_with_fixed_cout(self, input_imsize):
        channel_step = self._get_channel_step(input_imsize)

        if input_imsize <= 56:
            cin_range = list(range(160, 640 + channel_step, channel_step))
        elif input_imsize <= 112:
            cin_range = list(range(12, 320 + channel_step, channel_step))
        elif input_imsize <= 224:
            cin_range = []
        else:
            assert False

        if input_imsize <= 56:
            cout_range = [16, 640]
        elif input_imsize <= 112:
            cout_range = [16, 320]
        elif input_imsize <= 224:
            cout_range = []
        else:
            assert False

        return itertools.product(cin_range, cout_range)

    def _get_cin_cout_range(self, input_imsize):
        cin_cout_range = set()

        for cin in self._get_channel_range(input_imsize):
            cin_cout_range.add((cin, cin))
        for cin, cout in itertools.chain(
            self._get_cin_cout_range_with_fixed_cin(input_imsize),
            self._get_cin_cout_range_with_fixed_cout(input_imsize)
        ):
            cin_cout_range.add((cin, cout))

        cin_cout_range = sorted(list(cin_cout_range))
        return cin_cout_range

    def _get_samples_without_filter(self):
        for input_imsize in [7, 14, 28, 56, 112, 224]:
            for cin, cout in self._get_cin_cout_range(input_imsize):
                for stride in [1, 2]:
                    for kernel_size in [1, 3, 5]:
                        yield ["", "Conv", input_imsize, cin, cout, "", "", stride, kernel_size]


class OpExperimentConvSampler(Sampler):
    @staticmethod
    def get_sample_titles():
        return ConvSampler.get_sample_titles()

    def _get_samples_without_filter(self):
        for imsize in available_imsizes():
            for cin in sparse_channels_from_imsize(imsize):
                for stride, ksize in itertools.product(
                    [1, 2], [1, 3, 5, 7]
                ):
                    if ksize > imsize:
                        continue
                    return ["", "Conv", imsize, cin, cin, "", "", stride, ksize]
