import os
from utils.class_with_settings import ClassWithSettings
from collections import namedtuple
from typing import List, Optional

from .factory import *

ModelDetail = namedtuple("ModelDetail", [
    "model_path", "preprocess", "input_node", "output_node"
])


class MetaModelDetail:
    AVAILABLE_QUANTIZATIONS = {
        "pb": [""],
        "tflite": ["", "float16", "weight", "int"],
        "saved_model": [""],
    }

    def __init__(
            self,
            model_path: str, preprocess: Preprocess,
            input_node: str, output_node: str,
            available_model_formats: List[str]
    ):
        self.model_path = model_path
        self.preprocess = preprocess
        self.input_node = input_node
        self.output_node = output_node
        self.available_model_formats = available_model_formats
        self._onedrive_path = os.path.expanduser(
            "~/Microsoft/Shihao Han (FA Talent) - ChannelNas/models")

    def _get_model_path(self, model_format, quantization):
        if model_format == "pb":
            return "{}/pb/{}.pb".format(self._onedrive_path, self.model_path)

        elif model_format == "tflite":
            if quantization != "":
                quantization = "_{}_quant".format(quantization)
            return "{}/tflite/{}{}.tflite".format(self._onedrive_path, self.model_path, quantization)

        elif model_format == "saved_model":
            return "{}/saved_model/{}".format(self._onedrive_path, self.model_path)

    def _get_preprocess(self, model_format, quantization):
        return self.preprocess

    def get_model_detail(self, model_format, quantization) -> ModelDetail:
        assert model_format in self.AVAILABLE_QUANTIZATIONS
        assert quantization in self.AVAILABLE_QUANTIZATIONS[model_format]

        return ModelDetail(
            model_path=self._get_model_path(model_format, quantization),
            preprocess=self._get_preprocess(model_format, quantization),
            input_node=self.input_node,
            output_node=self.output_node
        )


meta_model_details = [
    # MetaModelDetail(
    #     "shufflenet_v1/shufflenet_v1_g3_1.5",
    #     shufflenet_preprocess,
    #     "input.1", "535",
    #     ["pb","tflite", "saved_model"]
    # ),
    # MetaModelDetail(
    #     "shufflenet_v1/shufflenet_v1_g8_1.0",
    #     shufflenet_preprocess,
    #     "input.1", "535",
    #     ["pb","tflite", "saved_model"]
    # ),
    # MetaModelDetail(
    #     "shufflenet_v2/shufflenet_v2_1.0",
    #     shufflenet_preprocess,
    #     "input.1", "626",
    #     ["pb","tflite", "saved_model"]
    # ),
    # MetaModelDetail(
    #     "shufflenet_v2/shufflenet_v2_1.5",
    #     shufflenet_preprocess,
    #     "input.1", "626",
    #     ["pb","tflite", "saved_model"]
    # ),
    MetaModelDetail(
        "mobilenet_v1/mobilenet_v1_1.0_224",
        inception_224_preprocess,
        "input", "MobilenetV1/Predictions/Reshape_1",
        ["pb", "tflite", "saved_model"]
    ),
    MetaModelDetail(
        "mobilenet_v2/mobilenet_v2_1.0_224",
        inception_224_preprocess,
        "input", "MobilenetV2/Predictions/Reshape_1",
        ["pb", "tflite", "saved_model"]
    ),
    MetaModelDetail(
        "mobilenet_v3/mobilenet_v3_large_224_1.0",
        inception_224_preprocess,
        "input", "MobilenetV3/Predictions/Softmax",
        ["pb", "tflite", "saved_model"]
    ),
    MetaModelDetail(
        "inception_v1/inception_v1",
        inception_224_preprocess,
        "input", "InceptionV1/Logits/Predictions/Reshape_1",
        ["pb", "tflite", "saved_model"]
    ),
    MetaModelDetail(
        "inception_v4/inception_v4",
        inception_299_preprocess,
        "input", "InceptionV4/Logits/Predictions",
        ["pb", "tflite", "saved_model"]
    ),
    MetaModelDetail(
        "mnasnet/mnasnet_a1",
        mnasnet_preprocess,
        "Placeholder", "logits",
        ["pb", "tflite", "saved_model"]
    ),
    MetaModelDetail(
        "nasnet/nasnet_a_mobile",
        inception_224_preprocess,
        "input", "final_layer/predictions",
        ["pb", "tflite", "saved_model"]
    ),
    MetaModelDetail(
        "proxyless/proxyless_mobile",
        proxyless_preprocess,
        "input_images", "classifier/linear/add",
        ["pb", "tflite", "saved_model"]
    ),
    MetaModelDetail(
        "proxyless/proxyless_mobile_14",
        proxyless_preprocess,
        "input_images", "classifier/linear/add",
        ["pb", "tflite", "saved_model"]
    ),
    MetaModelDetail(
        "resnet_v2/resnet_v2_50_299",
        inception_299_preprocess,
        "input", "resnet_v2_50/predictions/Reshape_1",
        ["pb", "tflite", "saved_model"]
    ),
    MetaModelDetail(
        "efficientnet/efficientnet_b0",
        efficientnet_b0_preprocess,
        "images", "Softmax",
        ["pb", "tflite", "saved_model"]
    ),
    MetaModelDetail(
        "efficientnet/efficientnet_b1",
        efficientnet_b1_preprocess,
        "images", "Softmax",
        ["pb", "tflite", "saved_model"]
    )
]


def get_model_details(
        model_names: Optional[List[str]],
        model_format: str,
        quantizations: List[str]
) -> List[ModelDetail]:
    ans = []
    for i in meta_model_details:
        if model_names is not None:
            skip = True
            for model_name in model_names:
                if model_name in i.model_path:
                    skip = False
            if skip:
                continue
        for quantization in quantizations:
            ans.append(i.get_model_detail(model_format, quantization))
    return ans
