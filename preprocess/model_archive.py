import os
from utils.class_with_settings import ClassWithSettings
from collections import namedtuple
from typing import List, Optional

from .factory import *

ModelDetail = namedtuple("ModelDetail", [
    "model_path", "preprocess", "input_node", "output_node"
])


class MetaModelDetail:
    AVAILABLE_VERSIONS = {
        "pb": ["", "patched"],
        "tflite": ["", "float16", "weight", "int", "edgetpu"],
        "saved_model": [""],
        "rknn": ["", "asymmetric_quantized_u8", "dynamic_fixed_point_8", "dynamic_fixed_point_16"]
    }

    def __init__(
            self,
            model_path: str, preprocess: Preprocess,
            input_node: str, output_node: str,
            available_model_formats: List[str],
            available_hardware: List[str],
    ):
        self.model_path = model_path
        self.preprocess = preprocess
        self.input_node = input_node
        self.output_node = output_node
        self.available_model_formats = available_model_formats
        self.available_hardware = available_hardware
        self._onedrive_path = os.path.expanduser(
            "~/Microsoft/Shihao Han (FA Talent) - ChannelNas/models")

    def _get_model_path(self, model_format, version):
        if model_format == "pb":
            if version != "":
                version = "_{}".format(version)
            return "{}/pb/{}{}.pb".format(self._onedrive_path, self.model_path, version)

        elif model_format == "tflite":
            if version in ["float16", "weight", "int"]:
                version = "_{}_quant".format(version)
            elif version in ["edgetpu"]:
                version = "_{}".format(version)
            return "{}/tflite/{}{}.tflite".format(self._onedrive_path, self.model_path, version)

        elif model_format == "saved_model":
            return "{}/saved_model/{}".format(self._onedrive_path, self.model_path)

        elif model_format == "rknn":
            if version != "":
                version = "_{}".format(version)
            return "{}/rknn/{}{}.rknn".format(self._onedrive_path, self.model_path, version)

    def _get_preprocess(self, model_format, version):
        if model_format == "rknn":
            return Preprocess({
                "preprocessor": self.preprocess.preprocessor,
                "func": "resize",
                "args": [np.float32 if version == "" else np.uint8]
            })
        else:
            return self.preprocess

    def get_model_detail(self, model_format, version) -> ModelDetail:
        assert model_format in self.AVAILABLE_VERSIONS
        assert version in self.AVAILABLE_VERSIONS[model_format]

        return ModelDetail(
            model_path=self._get_model_path(model_format, version),
            preprocess=self._get_preprocess(model_format, version),
            input_node=self.input_node,
            output_node=self.output_node
        )


meta_model_details = [
    MetaModelDetail(
        "shufflenet_v1/shufflenet_v1_g3_1.5",
        shufflenet_preprocess,
        "input.1", "535",
        ["pb", "tflite", "saved_model"],
        ["cpu"]
    ),
    MetaModelDetail(
        "shufflenet_v1/shufflenet_v1_g8_1.0",
        shufflenet_preprocess,
        "input.1", "535",
        ["pb", "tflite", "saved_model"],
        ["cpu"]
    ),
    MetaModelDetail(
        "shufflenet_v2/shufflenet_v2_1.0",
        shufflenet_preprocess,
        "input.1", "626",
        ["pb", "tflite", "saved_model"],
        ["cpu"]
    ),
    MetaModelDetail(
        "shufflenet_v2/shufflenet_v2_1.5",
        shufflenet_preprocess,
        "input.1", "626",
        ["pb", "tflite", "saved_model"],
        ["cpu"]
    ),
    MetaModelDetail(
        "mobilenet_v1/mobilenet_v1_1.0_224",
        inception_224_preprocess,
        "input", "MobilenetV1/Predictions/Reshape_1",
        ["pb", "tflite", "saved_model", "rknn"],
        ["cpu", "mobile_gpu", "rk", "edgetpu"]
    ),
    MetaModelDetail(
        "mobilenet_v2/mobilenet_v2_1.0_224",
        inception_224_preprocess,
        "input", "MobilenetV2/Predictions/Reshape_1",
        ["pb", "tflite", "saved_model", "rknn"],
        ["cpu", "mobile_gpu", "rk", "edgetpu"]
    ),
    MetaModelDetail(
        "mobilenet_v3/mobilenet_v3_large_224_1.0",
        inception_224_preprocess,
        "input", "MobilenetV3/Predictions/Softmax",
        ["pb", "tflite", "saved_model", "rknn"],
        ["cpu", "mobile_gpu", "rk"]
    ),
    MetaModelDetail(
        "inception_v1/inception_v1",
        inception_224_preprocess,
        "input", "InceptionV1/Logits/Predictions/Reshape_1",
        ["pb", "tflite", "saved_model", "rknn"],
        ["cpu", "mobile_gpu", "rk", "edgetpu"]
    ),
    MetaModelDetail(
        "inception_v4/inception_v4",
        inception_299_preprocess,
        "input", "InceptionV4/Logits/Predictions",
        ["pb", "tflite", "saved_model", "rknn"],
        ["cpu", "mobile_gpu", "rk", "edgetpu"]
    ),
    MetaModelDetail(
        "mnasnet/mnasnet_a1",
        mnasnet_preprocess,
        "Placeholder", "logits",
        ["pb", "tflite", "saved_model"],
        ["cpu", "edgetpu"]
    ),
    MetaModelDetail(
        "nasnet/nasnet_a_mobile",
        inception_224_preprocess,
        "input", "final_layer/predictions",
        ["pb", "tflite", "saved_model"],
        ["cpu", "edgetpu"]
    ),
    MetaModelDetail(
        "proxyless/proxyless_mobile",
        proxyless_preprocess,
        "input_images", "classifier/linear/add",
        ["pb", "tflite", "saved_model"],
        ["cpu", "mobile_gpu", "edgetpu"]
    ),
    MetaModelDetail(
        "proxyless/proxyless_mobile_14",
        proxyless_preprocess,
        "input_images", "classifier/linear/add",
        ["pb", "tflite", "saved_model"],
        ["cpu", "mobile_gpu", "edgetpu"]
    ),
    MetaModelDetail(
        "resnet_v2/resnet_v2_50_299",
        inception_299_preprocess,
        "input", "resnet_v2_50/predictions/Reshape_1",
        ["pb", "tflite", "saved_model", "rknn"],
        ["cpu", "mobile_gpu", "rk", "edgetpu"]
    ),
    MetaModelDetail(
        "efficientnet/efficientnet_b0",
        efficientnet_b0_preprocess,
        "images", "Softmax",
        ["pb", "tflite", "saved_model"],
        ["cpu"]
    ),
    MetaModelDetail(
        "efficientnet/efficientnet_b1",
        efficientnet_b1_preprocess,
        "images", "Softmax",
        ["pb", "tflite", "saved_model"],
        ["cpu"]
    )
]


def get_model_details(
        model_names: Optional[List[str]],
        model_format: str,
        versions: List[str],
        hardware: str = "cpu"
) -> List[ModelDetail]:
    ans = []
    for i in filter(lambda i: hardware in i.available_hardware, meta_model_details):
        if model_names is not None:
            skip = True
            for model_name in model_names:
                if model_name in i.model_path:
                    skip = False
            if skip:
                continue
        if not (model_format in i.available_model_formats):
            continue
        for version in versions:
            ans.append(i.get_model_detail(model_format, version))
    return ans
