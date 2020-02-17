from .preprocess import Preprocess

from .torch_preprocessor import TorchPreprocessor
from .tf_preprocessor import TfPreprocessor

import numpy as np
import os
from collections import namedtuple

# shufflenet:
# https://github.com/megvii-model/ShuffleNet-Series
shufflenet_preprocess = Preprocess({
    "preprocessor": TorchPreprocessor({
        "imsize": 224,
        "use_opencv_resize": True,
        "use_bgr": True,
        "use_normalization": False,
        "use_nhwc": False
    })
})

# mobilenet v1, mobilenet v2, mobilenet v3 large, inception v1, nasnet a mobile:
# https://github.com/tensorflow/models/tree/master/research/slim
inception_224_preprocess = Preprocess({
    "preprocessor": TfPreprocessor({
        "imsize": 224,
        "use_crop_padding": False,
        "resize_func": "resize_bilinear",
        "use_inception": True,
    })
})

# inception v4, resnet v2 50:
# https://github.com/tensorflow/models/tree/master/research/slim
inception_299_preprocess = Preprocess({
    "preprocessor": TfPreprocessor({
        "imsize": 299,
        "use_crop_padding": False,
        "resize_func": "resize_bilinear",
        "use_inception": True,
    })
})

# mnasnet:
# https://github.com/tensorflow/tpu/tree/master/models/official/mnasnet
mnasnet_preprocess = Preprocess({
    "preprocessor": TfPreprocessor({
        "imsize": 224,
        "use_crop_padding": True,
        "resize_func": "resize_bilinear",
    }),
    "func": "resize",
    "args": [np.float32]
})

# proxyless nas net:
# https://github.com/mit-han-lab/proxylessnas
proxyless_preprocess = Preprocess({
    "preprocessor": TorchPreprocessor({
        "imsize": 224,
        "use_opencv_resize": False,
        "use_bgr": False,
        "use_normalization": True,
        "use_nhwc": True
    })
})

# efficientnet b0: https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet
efficientnet_b0_preprocess = Preprocess({
    "preprocessor": TfPreprocessor({
        "imsize": 224,
        "use_crop_padding": True,
        "resize_func": "resize_bicubic",
        "use_inception": True,
    })
})

# efficientnet b1: https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet
efficientnet_b1_preprocess = Preprocess({
    "preprocessor": TfPreprocessor({
        "imsize": 240,
        "use_crop_padding": True,
        "resize_func": "resize_bicubic",
        "use_inception": True,
    })
})


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
        _onedrive_path + "/tflite/efficientnet/efficientnet_b0.tflite",
        efficientnet_b0_preprocess
    ),
    ModelDetail(
        _onedrive_path + "/tflite/efficientnet/efficientnet_b1.tflite",
        efficientnet_b1_preprocess
    )
]
