from .preprocess import Preprocessor
import numpy as np


class TfPreprocessor(Preprocessor):
    CROP_PADDING = 32
    CENTRAL_FRACTION = 0.875
    MEAN_RGB = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    STDDEV_RGB = [0.229 * 255, 0.224 * 255, 0.225 * 255]

    @staticmethod
    def default_settings():
        return {
            **Preprocessor.default_settings(),
            "use_crop_padding": True,
            "resize_func": "resize_bicubic",
            "use_inception": True,
        }

    def get_normalization_parameter(self):
        if self.settings["use_inception"]:
            return [[127.5, 127.5, 127.5], [127.5, 127.5, 127.5]]
        else:
            return [self.MEAN_RGB, self.STDDEV_RGB]

    def _imagenet_accuracy_eval_flags(self):
        return {
            "use_crop_padding": self.settings["use_crop_padding"],
        }

    def __init__(self, settings={}):
        super().__init__(settings)
        import tensorflow as tf

        self.use_crop_padding = self.settings["use_crop_padding"]
        if self.settings["resize_func"] == "resize_bicubic":
            self.resize_func = tf.compat.v1.image.resize_bicubic
        elif self.settings["resize_func"] == "resize_bilinear":
            self.resize_func = tf.compat.v1.image.resize_bilinear
        else:
            assert False
        self.use_inception = self.settings["use_inception"]

        self._construct_tf_graph()

    def _construct_tf_graph(self):
        import tensorflow as tf

        with tf.Graph().as_default() as graph:
            self.image_bytes = tf.compat.v1.placeholder(tf.string)
            self.ret_imread = tf.image.decode_jpeg(
                self.image_bytes, channels=3)

            if self.use_crop_padding:
                shape = tf.image.extract_jpeg_shape(self.image_bytes)
                image_height = shape[0]
                image_width = shape[1]
                padded_center_crop_size = tf.cast(
                    ((self.imsize / (self.imsize + self.CROP_PADDING)) *
                     tf.cast(tf.minimum(image_height, image_width), tf.float32)),
                    tf.int32)
                offset_height = (
                    (image_height - padded_center_crop_size) + 1) // 2
                offset_width = (
                    (image_width - padded_center_crop_size) + 1) // 2
                crop_window = tf.stack([
                    offset_height, offset_width,
                    padded_center_crop_size, padded_center_crop_size
                ])
                image = tf.image.decode_and_crop_jpeg(
                    self.image_bytes, crop_window, channels=3)
            else:
                image = tf.image.central_crop(
                    self.ret_imread,
                    central_fraction=self.CENTRAL_FRACTION
                )

            self.ret_resize = self.resize_func(
                [image], [self.imsize, self.imsize])[0]

            if self.use_inception:
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
