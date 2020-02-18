import os
from collections import namedtuple

from .factory import *

ModelDetail = namedtuple("ModelDetail", [
    "model_path", "preprocess", "input_node", "output_node"
])

_onedrive_path = os.path.expanduser(
    "~/Microsoft/Shihao Han (FA Talent) - ChannelNas/models")


tflite_model_details = [
    ModelDetail(
        _onedrive_path + "/tflite/shufflenet_v1/shufflenet_v1_g3_1.5.tflite",
        shufflenet_preprocess, "input.1", "535"
    ),
    ModelDetail(
        _onedrive_path + "/tflite/shufflenet_v1/shufflenet_v1_g8_1.0.tflite",
        shufflenet_preprocess, "input.1", "535"
    ),
    ModelDetail(
        _onedrive_path + "/tflite/shufflenet_v2/shufflenet_v2_1.0.tflite",
        shufflenet_preprocess, "input.1", "626"
    ),
    ModelDetail(
        _onedrive_path + "/tflite/shufflenet_v2/shufflenet_v2_1.5.tflite",
        shufflenet_preprocess, "input.1", "626"
    ),
    ModelDetail(
        _onedrive_path + "/tflite/mobilenet_v1/mobilenet_v1_1.0_224.tflite",
        inception_224_preprocess, "input", "MobilenetV1/Predictions/Reshape_1"
    ),
    ModelDetail(
        _onedrive_path + "/tflite/mobilenet_v2/mobilenet_v2_1.0_224.tflite",
        inception_224_preprocess, "input", "MobilenetV2/Predictions/Reshape_1"
    ),
    ModelDetail(
        _onedrive_path + "/tflite/mobilenet_v3/mobilenet_v3_large_224_1.0.tflite",
        inception_224_preprocess, "input", "MobilenetV3/Predictions/Softmax"
    ),
    ModelDetail(
        _onedrive_path + "/tflite/inception_v1/inception_v1.tflite",
        inception_224_preprocess, "input", "InceptionV1/Logits/Predictions/Reshape_1"
    ),
    ModelDetail(
        _onedrive_path + "/tflite/inception_v4/inception_v4.tflite",
        inception_299_preprocess, "input", "InceptionV4/Logits/Predictions"
    ),
    ModelDetail(
        _onedrive_path + "/tflite/mnasnet/mnasnet_a1.tflite",
        mnasnet_preprocess, "Placeholder", "logits"
    ),
    ModelDetail(
        _onedrive_path + "/tflite/nasnet/nasnet_a_mobile.tflite",
        inception_224_preprocess, "input", "final_layer/predictions"
    ),
    ModelDetail(
        _onedrive_path + "/tflite/proxyless/proxyless_mobile.tflite",
        proxyless_preprocess, "input_images", "classifier/linear/add"
    ),
    ModelDetail(
        _onedrive_path + "/tflite/proxyless/proxyless_mobile_14.tflite",
        proxyless_preprocess, "input_images", "classifier/linear/add"
    ),
    ModelDetail(
        _onedrive_path + "/tflite/resnet_v2/resnet_v2_50_299.tflite",
        inception_299_preprocess, "input", "resnet_v2_50/predictions/Reshape_1"
    ),
    ModelDetail(
        _onedrive_path + "/tflite/efficientnet/efficientnet_b0.tflite",
        efficientnet_b0_preprocess, "images", "Softmax"
    ),
    ModelDetail(
        _onedrive_path + "/tflite/efficientnet/efficientnet_b1.tflite",
        efficientnet_b1_preprocess, "images", "Softmax"
    )
]
