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

    def get_normalization_parameter(self):
        """Get normalization parameter
        Returns:
            [MEAN_RGB, STD_RGB]
        """
        ...

    def _imagenet_accuracy_eval_flags(self):
        ...

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

    def __init__(self, settings={}):
        super().__init__(settings)
        self.preprocessor: Preprocess = self.settings["preprocessor"]

    def get_imsize(self):
        return self.preprocessor.settings["imsize"]

    def get_normalization_parameter(self):
        normalization_parameter = self.preprocessor.get_normalization_parameter()
        if self.settings["func"] == "resize":
            normalization_parameter = [[0] * 3, [1] * 3]
        return normalization_parameter

    def imagenet_accuracy_eval_flags(self):
        ret = self.preprocessor._imagenet_accuracy_eval_flags()
        normalization_parameter = self.get_normalization_parameter()

        return {
            **ret,
            "mean": ','.join(map(str, normalization_parameter[0])),
            "scale": ','.join(map(lambda v: str(1 / v), normalization_parameter[1]))
        }

    def execute(self, image_path: str):
        return getattr(
            self.settings["preprocessor"],
            self.settings["func"])(
            image_path,
            *self.settings["args"]
        )
