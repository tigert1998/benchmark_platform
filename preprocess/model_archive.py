import os
from collections import namedtuple

from .factory import *

ModelDetail = namedtuple("ModelDetail", ["model_path", "preprocess"])

_onedrive_path = os.path.expanduser(
    "~/Microsoft/Shihao Han (FA Talent) - ChannelNas/models")


tflite_model_details = [
    ModelDetail(
        _onedrive_path + "/tflite/shufflenet_v1/shufflenet_v1_g3_1.5.tflite",
        shufflenet_preprocess
    ),
    ModelDetail(
        _onedrive_path + "/tflite/shufflenet_v1/shufflenet_v1_g8_1.0.tflite",
        shufflenet_preprocess
    ),
    ModelDetail(
        _onedrive_path + "/tflite/shufflenet_v2/shufflenet_v2_1.0.tflite",
        shufflenet_preprocess
    ),
    ModelDetail(
        _onedrive_path + "/tflite/shufflenet_v2/shufflenet_v2_1.5.tflite",
        shufflenet_preprocess
    ),
    ModelDetail(
        _onedrive_path + "/tflite/mobilenet_v1/mobilenet_v1_1.0_224.tflite",
        inception_224_preprocess
    ),
    ModelDetail(
        _onedrive_path + "/tflite/mobilenet_v2/mobilenet_v2_1.0_224.tflite",
        inception_224_preprocess
    ),
    ModelDetail(
        _onedrive_path + "/tflite/mobilenet_v3/mobilenet_v3_large_224_1.0.tflite",
        inception_224_preprocess
    ),
    ModelDetail(
        _onedrive_path + "/tflite/inception_v1/inception_v1.tflite",
        inception_224_preprocess
    ),
    ModelDetail(
        _onedrive_path + "/tflite/inception_v4/inception_v4.tflite",
        inception_299_preprocess
    ),
    ModelDetail(
        _onedrive_path + "/tflite/mnasnet_a1/mnasnet_a1.tflite",
        mnasnet_preprocess
    ),
    ModelDetail(
        _onedrive_path + "/tflite/nasnet_a_mobile/nasnet_a_mobile.tflite",
        inception_224_preprocess
    ),
    ModelDetail(
        _onedrive_path + "/tflite/proxyless/proxyless_mobile.tflite",
        proxyless_preprocess
    ),
    ModelDetail(
        _onedrive_path + "/tflite/proxyless/proxyless_mobile_14.tflite",
        proxyless_preprocess
    ),
    ModelDetail(
        _onedrive_path + "/tflite/resnet_v2/resnet_v2_50_299.tflite",
        inception_299_preprocess
    ),
    ModelDetail(
        _onedrive_path + "/tflite/efficientnet/efficientnet_b0.tflite",
        efficientnet_b0_preprocess
    ),
    ModelDetail(
        _onedrive_path + "/tflite/efficientnet/efficientnet_b1.tflite",
        efficientnet_b1_preprocess
    )
]
