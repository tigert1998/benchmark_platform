from collections import namedtuple

ShuffleNetBlock = namedtuple(
    'ShuffleNetBlock', ["output_size", "stride", "repeat", "output_channels"])

# g = 3, x1.5
shufflenetv1_stages = [
    [
        ShuffleNetBlock(28, 2, 1, 360),
        ShuffleNetBlock(28, 1, 3, 360)
    ],
    [
        ShuffleNetBlock(14, 2, 1, 720),
        ShuffleNetBlock(14, 1, 7, 720)
    ],
    [
        ShuffleNetBlock(7, 2, 1, 1440),
        ShuffleNetBlock(7, 1, 3, 1440)
    ]
]

# x1.5
shufflenetv2_stages = [
    [
        ShuffleNetBlock(28, 2, 1, 176),
        ShuffleNetBlock(28, 1, 3, 176)
    ],
    [
        ShuffleNetBlock(14, 2, 1, 352),
        ShuffleNetBlock(14, 1, 7, 352)
    ],
    [
        ShuffleNetBlock(7, 2, 1, 704),
        ShuffleNetBlock(7, 1, 3, 704)
    ]
]


def align(n: int, divisor: int):
    assert isinstance(n, int) and isinstance(divisor, int)
    return ((n + divisor - 1) // divisor) * divisor


def merge_profiles(profiles):
    dic = {}
    for i in profiles:
        names = i[-1]
        assert(isinstance(i, list))
        assert(isinstance(names, list))
        # for hashable
        key = tuple(i[:-1])
        if dic.get(key) is None:
            dic[key] = names
        else:
            dic[key] = names + dic[key]
    res = []
    for key, value in sorted(dic.items()):
        res.append(list(key) + [value])
    return res


def op_name_to_model_name(op_name: str):
    if 'mobilenetv2' in op_name:
        return 'MobileNetV2'
    elif 'shufflenetv1' in op_name:
        return 'ShuffleNetV1'
    elif 'shufflenetv2' in op_name:
        return 'ShuffleNetV2'
    else:
        assert False


# the followings helpers are for the new tests

def sparse_channels_from_imsize(imsize: int):
    if imsize in [224, 112, 56]:
        return [32, 64, 96]
    elif imsize in [28]:
        return [32, 64, 96, 128]
    elif imsize in [14]:
        return [160, 240, 256, 320]
    elif imsize in [7]:
        return [160, 240, 320, 512]
    elif imsize in [3, 1]:
        return [160, 240, 320]
    else:
        assert False


def available_imsizes():
    return [1, 3, 7, 14, 28, 56, 112, 224]


def available_num_groups():
    return [2, 3, 4, 8]
