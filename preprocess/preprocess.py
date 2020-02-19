from utils.class_with_settings import ClassWithSettings

import numpy as np


def default_central_crop(image: np.ndarray, crop_size: int) -> np.ndarray:
    height, width, _ = image.shape
    offset_height = ((height - crop_size + 1) // 2)
    offset_width = ((width - crop_size + 1) // 2)
    return image[
        offset_height: offset_height + crop_size,
        offset_width: offset_width + crop_size,
        :
    ]


class Preprocessor(ClassWithSettings):
    @staticmethod
    def default_settings():
        return {
            **ClassWithSettings.default_settings(),
            "imsize": 224
        }

    def imagenet_accuracy_eval_flags(self):
        return {}

    def __init__(self, settings={}):
        super().__init__(settings)
        self.imsize = self.settings["imsize"]

    @staticmethod
    def imread(image_path: str) -> np.ndarray:
        import cv2
        return cv2.imread(image_path)[:, :, ::-1]

    def _resize(self, image: np.ndarray, dtype) -> np.ndarray:
        ...

    def resize(self, image_path: str, dtype) -> np.ndarray:
        image = self.imread(image_path)
        _, _, channels = image.shape
        assert 3 == channels
        return self._resize(image, dtype)

    def _apply_mean_and_scale(self, image: np.ndarray) -> np.ndarray:
        ...

    def preprocess(self, image_path: str, dtype=np.float32) -> np.ndarray:
        image = self.resize(image_path, np.float32)
        image = self._apply_mean_and_scale(image)
        return image.astype(dtype)


class Preprocess(ClassWithSettings):
    @staticmethod
    def default_settings():
        return {
            **ClassWithSettings.default_settings(),
            "preprocessor": Preprocessor(),
            "func": "preprocess",
            "args": []
        }

    def imagenet_accuracy_eval_flags(self):
        ret = self.settings["preprocessor"].imagenet_accuracy_eval_flags()
        if self.settings["func"] == "resize":
            ret = {**ret, "mean": "0,0,0", "scale": "1,1,1"}
        return ret

    def execute(self, image_path: str):
        return getattr(
            self.settings["preprocessor"],
            self.settings["func"])(
            image_path,
            *self.settings["args"]
        )
