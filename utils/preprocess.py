import numpy as np
import cv2


class Preprocess:
    @classmethod
    def imread(cls, image_path: str) -> np.ndarray:
        return cv2.imread(image_path)[:, :, ::-1]

    @classmethod
    def _resize(cls, image: np.ndarray, imsize: int, dtype) -> np.ndarray:
        ...

    @classmethod
    def resize(cls, image_path: str, imsize: int, dtype) -> np.ndarray:
        image = cls.imread(image_path)
        _, _, channels = image.shape
        assert 3 == channels
        return cls._resize(image, imsize, dtype)

    @classmethod
    def _apply_mean_and_scale(cls, image: np.ndarray) -> np.ndarray:
        ...

    @classmethod
    def preprocess(cls, image_path: str, imsize: int, dtype=np.float32) -> np.ndarray:
        image = cls.resize(image_path, imsize, np.float32)
        image = cls.apply_mean_and_scale(image)
        return image.astype(dtype)

    @classmethod
    def central_crop(cls, image: np.ndarray, crop_size: int) -> np.ndarray:
        height, width, _ = image.shape
        offset_height = ((height - crop_size + 1) // 2)
        offset_width = ((width - crop_size + 1) // 2)
        return image[
            offset_height: offset_height + crop_size,
            offset_width: offset_width + crop_size,
            :
        ]


class InceptionPreprocess(Preprocess):
    CROPPING_FRACTION = 0.875

    @classmethod
    def _resize(cls, image: np.ndarray, imsize: int, dtype) -> np.ndarray:
        height, width, _ = image.shape
        crop_size = int(min(height, width) * cls.CROPPING_FRACTION)
        image = Preprocess.central_crop(image, crop_size)
        image = cv2.resize(
            image.astype(np.float32), (imsize, imsize),
            interpolation=cv2.INTER_CUBIC
        ).astype(dtype)
        return np.expand_dims(image, 0)

    @classmethod
    def _apply_mean_and_scale(cls, image: np.ndarray) -> np.ndarray:
        return (image * 2.0 / 255 - 1.0).astype(np.float32)


class InceptionPreprocessTF(Preprocess):
    CROP_PADDING = 32

    @classmethod
    def _construct_tf_graph(cls, imsize: int):
        if getattr(cls, "current_tf_graph_imsize", None) == imsize:
            return
        else:
            cls.current_tf_graph_imsize = imsize

        import tensorflow as tf

        with tf.Graph().as_default() as graph:
            cls.image_bytes = tf.placeholder(tf.string)
            cls.ret_imread = tf.image.decode_jpeg(cls.image_bytes, channels=3)

            shape = tf.image.extract_jpeg_shape(cls.image_bytes)
            image_height = shape[0]
            image_width = shape[1]

            padded_center_crop_size = tf.cast(
                ((imsize / (imsize + cls.CROP_PADDING)) *
                 tf.cast(tf.minimum(image_height, image_width), tf.float32)),
                tf.int32)

            offset_height = ((image_height - padded_center_crop_size) + 1) // 2
            offset_width = ((image_width - padded_center_crop_size) + 1) // 2
            crop_window = tf.stack([
                offset_height, offset_width,
                padded_center_crop_size, padded_center_crop_size
            ])
            image = tf.image.decode_and_crop_jpeg(
                cls.image_bytes, crop_window, channels=3)
            cls.ret_resize = tf.image.resize_bicubic(
                [image], [imsize, imsize])[0]
            cls.ret_preprocess = cls.ret_resize * 2.0 / 255 - 1.0

        cls.sess = tf.Session(graph=graph)

    @classmethod
    def _run_op(cls, op_name, image_path: str, imsize: int, dtype) -> np.ndarray:
        cls._construct_tf_graph(imsize)
        with open(image_path, "rb") as f:
            return cls.sess.run(getattr(cls, op_name), feed_dict={
                cls.image_bytes: f.read()
            }).astype(dtype)

    @classmethod
    def imread(cls, image_path: str) -> np.ndarray:
        return cls._run_op("ret_imread", image_path, 224, np.uint8)

    @classmethod
    def resize(cls, image_path: str, imsize: int, dtype=np.uint8) -> np.ndarray:
        return np.expand_dims(cls._run_op("ret_resize", image_path, imsize, dtype), 0)

    @classmethod
    def preprocess(cls, image_path: str, imsize: int, dtype=np.float32) -> np.ndarray:
        return np.expand_dims(cls._run_op("ret_preprocess", image_path, imsize, dtype), 0)
