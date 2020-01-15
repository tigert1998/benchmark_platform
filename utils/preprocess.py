import numpy as np
import cv2


class Preprocess:
    def __init__(self, imsize: int):
        self.imsize = imsize

    @staticmethod
    def imread(image_path: str) -> np.ndarray:
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

    @staticmethod
    def central_crop(image: np.ndarray, crop_size: int) -> np.ndarray:
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

    def _resize(self, image: np.ndarray, dtype) -> np.ndarray:
        height, width, _ = image.shape
        crop_size = int(min(height, width) * self.CROPPING_FRACTION)
        image = Preprocess.central_crop(image, crop_size)
        image = cv2.resize(
            image.astype(np.float32), (self.imsize, self.imsize),
            interpolation=cv2.INTER_CUBIC
        ).astype(dtype)
        return np.expand_dims(image, 0)

    def _apply_mean_and_scale(self, image: np.ndarray) -> np.ndarray:
        return (image * 2.0 / 255 - 1.0).astype(np.float32)


class TPURepoPreprocess(Preprocess):
    CROP_PADDING = 32
    MEAN_RGB = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    STDDEV_RGB = [0.229 * 255, 0.224 * 255, 0.225 * 255]

    def __init__(self, imsize: int, is_inception: bool):
        super().__init__(imsize)
        self.is_inception = is_inception
        self._construct_tf_graph()

    def _construct_tf_graph(self):
        import tensorflow as tf

        with tf.Graph().as_default() as graph:
            self.image_bytes = tf.compat.v1.placeholder(tf.string)
            self.ret_imread = tf.image.decode_jpeg(
                self.image_bytes, channels=3)

            shape = tf.image.extract_jpeg_shape(self.image_bytes)
            image_height = shape[0]
            image_width = shape[1]

            padded_center_crop_size = tf.cast(
                ((self.imsize / (self.imsize + self.CROP_PADDING)) *
                 tf.cast(tf.minimum(image_height, image_width), tf.float32)),
                tf.int32)

            offset_height = ((image_height - padded_center_crop_size) + 1) // 2
            offset_width = ((image_width - padded_center_crop_size) + 1) // 2
            crop_window = tf.stack([
                offset_height, offset_width,
                padded_center_crop_size, padded_center_crop_size
            ])
            image = tf.image.decode_and_crop_jpeg(
                self.image_bytes, crop_window, channels=3)
            self.ret_resize = tf.image.resize_bicubic(
                [image], [self.imsize, self.imsize])[0]

            if self.is_inception:
                self.ret_preprocess = self.ret_resize * 2.0 / 255 - 1.0
            else:
                self.ret_preprocess = (
                    self.ret_resize -
                    tf.constant(
                        self.MEAN_RGB,
                        shape=[1, 1, 3],
                        dtype=tf.float32
                    )
                ) / tf.constant(
                    self.STDDEV_RGB,
                    shape=[1, 1, 3],
                    dtype=tf.float32
                )

        self.sess = tf.compat.v1.Session(graph=graph)

    def _run_op(self, op, image_path: str, dtype) -> np.ndarray:
        with open(image_path, "rb") as f:
            return self.sess.run(op, feed_dict={
                self.image_bytes: f.read()
            }).astype(dtype)

    def imread(self, image_path: str) -> np.ndarray:
        return self._run_op(self.ret_imread, image_path, np.uint8)

    def resize(self, image_path: str, dtype=np.uint8) -> np.ndarray:
        return np.expand_dims(self._run_op(self.ret_resize, image_path, dtype), 0)

    def preprocess(self, image_path: str, dtype=np.float32) -> np.ndarray:
        return np.expand_dims(self._run_op(self.ret_preprocess, image_path, dtype), 0)
