from .preprocess import Preprocess

from .torch_preprocessor import TorchPreprocessor
from .tf_preprocessor import TfPreprocessor

import numpy as np

# shufflenet:
# https://github.com/megvii-model/ShuffleNet-Series
shufflenet_preprocess = Preprocess({
    "preprocessor": TorchPreprocessor({
        "imsize": 224,
        "use_opencv_resize": True,
        "use_normalization": False,
    })
})

# mobilenet v1, mobilenet v2, mobilenet v3 large, inception v1, nasnet a mobile:
# https://github.com/tensorflow/models/tree/master/research/slim
inception_224_preprocess = Preprocess({
    "preprocessor": TfPreprocessor({
        "imsize": 224,
        "use_crop_padding": False,
        "resize_func": "resize_bilinear",
        "normalization": "inception",
    })
})

# inception v4, resnet v2 50:
# https://github.com/tensorflow/models/tree/master/research/slim
inception_299_preprocess = Preprocess({
    "preprocessor": TfPreprocessor({
        "imsize": 299,
        "use_crop_padding": False,
        "resize_func": "resize_bilinear",
        "normalization": "inception",
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
        "use_normalization": True,
    })
})

# efficientnet b0: https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet
efficientnet_b0_preprocess = Preprocess({
    "preprocessor": TfPreprocessor({
        "imsize": 224,
        "use_crop_padding": True,
        "resize_func": "resize_bicubic",
        "normalization": "inception",
    })
})

# efficientnet b1: https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet
efficientnet_b1_preprocess = Preprocess({
    "preprocessor": TfPreprocessor({
        "imsize": 240,
        "use_crop_padding": True,
        "resize_func": "resize_bicubic",
        "normalization": "inception",
    })
})

# resnet v1: https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v1.py
resnet_v1_preprocess = Preprocess({
    "preprocessor": TfPreprocessor({
        "imsize": 224,
        "use_crop_padding": True,
        "resize_func": "resize_bilinear",
        "normalization": "vgg",
    })
})
