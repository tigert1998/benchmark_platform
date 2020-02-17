from .preprocess import Preprocessor

import numpy as np
import PIL
from PIL import Image
import torch


class OpencvResize:
    def __init__(self, size: int):
        self.size = size

    def __call__(self, img):
        import cv2
        img = np.asarray(img)  # (H,W,3) RGB
        img = img[:, :, ::-1]  # 2 BGR
        img = np.ascontiguousarray(img)
        H, W, _ = img.shape
        target_size = (int(self.size/H * W + 0.5),
                       self.size) if H < W else (self.size, int(self.size/W * H + 0.5))
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
        img = img[:, :, ::-1]  # 2 RGB
        img = np.ascontiguousarray(img)
        img = Image.fromarray(img)
        return img


class ToBGRTensor:
    def __call__(self, img):
        assert isinstance(img, (np.ndarray, PIL.Image.Image))
        if isinstance(img, PIL.Image.Image):
            img = np.asarray(img)
        img = img[:, :, ::-1]  # 2 BGR
        img = np.transpose(img, [2, 0, 1])  # 2 (3, H, W)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).float()
        return img


class TorchPreprocessor(Preprocessor):
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    @staticmethod
    def default_settings():
        return {
            **Preprocessor.default_settings(),
            "use_opencv_resize": False,
            "use_bgr": False,
            "use_normalization": True,
            "use_nhwc": True
        }

    def _run_until(self, idx: int, image_path: str, dtype) -> np.ndarray:
        last = image_path
        for i in range(idx + 1):
            last = self.transform[i](last)
        return np.ascontiguousarray(last, dtype=dtype)

    def imread(self, image_path: str) -> np.ndarray:
        return self._run_until(self.imread_idx, image_path, np.uint8)

    def resize(self, image_path: str, dtype=np.uint8) -> np.ndarray:
        return np.expand_dims(self._run_until(self.resize_idx, image_path, dtype), 0)

    def preprocess(self, image_path: str, dtype=np.float32) -> np.ndarray:
        return np.expand_dims(self._run_until(self.preprocess_idx, image_path, dtype), 0)

    def __init__(self, settings={}):
        super().__init__(settings)

        from torchvision import transforms

        def _imread(image_path: str):
            with open(image_path, 'rb') as f:
                return Image.open(f).convert('RGB')

        self.transform = [_imread]
        self.imread_idx = 0

        if self.settings["use_opencv_resize"]:
            self.transform.append(OpencvResize(256))
        else:
            self.transform.append(transforms.Resize(256))

        self.transform.append(transforms.CenterCrop(self.settings["imsize"]))

        if self.settings["use_bgr"]:
            self.transform.append(ToBGRTensor())
            self.resize_idx = 3
        else:
            self.resize_idx = 2
            self.transform.append(transforms.ToTensor())

        if self.settings["use_normalization"]:
            self.transform.append(transforms.Normalize(
                mean=self.MEAN,
                std=self.STD
            ))
        if self.settings["use_nhwc"]:
            self.transform.append(lambda tensor: tensor.permute(1, 2, 0))

        self.preprocess_idx = len(self.transform) - 1
