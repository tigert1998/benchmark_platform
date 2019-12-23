import numpy as np
import cv2


class Preprocess:
    @classmethod
    def resize(cls, image: np.ndarray, imsize: int) -> np.ndarray:
        pass

    @classmethod
    def apply_mean_and_scale(cls, image: np.ndarray) -> np.ndarray:
        pass

    @classmethod
    def preprocess(cls, image: np.ndarray, imsize: int) -> np.ndarray:
        _, _, channels = image.shape
        assert 3 == channels
        image = cls.resize(image, imsize)
        image = cls.apply_mean_and_scale(image)
        return np.reshape(image, (1, imsize, imsize, 3))

    @classmethod
    def central_crop(cls, image: np.ndarray, crop_size: int) -> np.ndarray:
        height, width, _ = image.shape
        offset_height = round((height - crop_size) / 2)
        offset_width = round((width - crop_size) / 2)
        return image[
            offset_height: offset_height + crop_size,
            offset_width: offset_width + crop_size,
            :
        ]


class VggPreprocess(Preprocess):
    RESIZE_SIDE = 256
    R_MEAN = 123.68
    G_MEAN = 116.779
    B_MEAN = 103.939

    @classmethod
    def aspect_preserving_resize(cls, image: np.ndarray, resize_side: int) -> np.ndarray:
        height, width, _ = image.shape
        scale = resize_side / min(height, width)
        new_height = round(height * scale)
        new_width = round(width * scale)
        return cv2.resize(
            image, (new_height, new_width),
            interpolation=cv2.INTER_LINEAR
        )

    @classmethod
    def resize(cls, image: np.ndarray, imsize: int) -> np.ndarray:
        image = cls.aspect_preserving_resize(image, cls.RESIZE_SIDE)
        return Preprocess.central_crop(image, imsize).astype(np.uint8)

    @classmethod
    def apply_mean_and_scale(cls, image: np.ndarray) -> np.ndarray:
        return (image - [
            cls.R_MEAN,
            cls.G_MEAN,
            cls.B_MEAN
        ]).astype(np.float32)


class InceptionPreprocess(Preprocess):
    CROPPING_FRACTION = 0.875

    @classmethod
    def resize(cls, image: np.ndarray, imsize: int) -> np.ndarray:
        height, width, _ = image.shape
        crop_size = round(min(height, width) * cls.CROPPING_FRACTION)
        image = Preprocess.central_crop(image, crop_size)
        return cv2.resize(
            image, (imsize, imsize),
            interpolation=cv2.INTER_LINEAR
        ).astype(np.uint8)

    @classmethod
    def apply_mean_and_scale(cls, image: np.ndarray) -> np.ndarray:
        return ((image - 127.5) / 127.5).astype(np.float32)
