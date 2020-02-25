from .test_single_layer import TestSingleLayer

import tensorflow as tf

from network.building_ops import channel_shuffle


class TestAdd(TestSingleLayer):
    def _generate_model(self, sample):
        model_path = "model"

        _, input_imsize, cin = sample

        tf.reset_default_graph()
        first_input_im = tf.placeholder(
            name="1st_input_im", dtype=tf.float32,
            shape=(1, input_imsize, input_imsize, cin))
        second_input_im = tf.placeholder(
            name="2nd_input_im", dtype=tf.float32,
            shape=(1, input_imsize, input_imsize, cin))

        self.inference_sdk.generate_model(
            model_path, [first_input_im, second_input_im], [first_input_im + second_input_im])
        return model_path, [first_input_im.get_shape().as_list()] * 2


class TestConcat(TestSingleLayer):
    def _generate_model(self, sample):
        model_path = "model"

        _, input_imsize, first_cin, second_cin = sample

        tf.reset_default_graph()
        first_input_im = tf.placeholder(
            name="1st_input_im", dtype=tf.float32,
            shape=(1, input_imsize, input_imsize, first_cin))
        second_input_im = tf.placeholder(
            name="2nd_input_im", dtype=tf.float32,
            shape=(1, input_imsize, input_imsize, second_cin))
        net = tf.concat([first_input_im, second_input_im], axis=-1)

        self.inference_sdk.generate_model(
            model_path, [first_input_im, second_input_im], [net])
        return model_path, [first_input_im.get_shape().as_list(), second_input_im.get_shape().as_list()]


class TestGlobalPooling(TestSingleLayer):
    def _generate_model(self, sample):
        model_path = "model"

        _, input_imsize, cin = sample

        tf.reset_default_graph()
        input_im = tf.placeholder(
            name="input_im", dtype=tf.float32,
            shape=(1, input_imsize, input_imsize, cin))
        net = tf.keras.layers.GlobalAveragePooling2D()(input_im)

        self.inference_sdk.generate_model(
            model_path, [input_im], [net])
        return model_path, [input_im.get_shape().as_list()]


class TestShuffle(TestSingleLayer):
    def _generate_model(self, sample):
        model_path = "model"

        _, input_imsize, cin, num_groups = sample

        tf.reset_default_graph()
        input_im = tf.placeholder(
            name="input_im", dtype=tf.float32,
            shape=(1, input_imsize, input_imsize, cin))
        net = channel_shuffle(input_im, num_groups)

        self.inference_sdk.generate_model(
            model_path, [input_im], [net])
        return model_path, [input_im.get_shape().as_list()]
