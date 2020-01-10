from .data_preparer_def import DataPreparerDef
from utils.utils import rm_ext
from utils.connection import Connection

import os

import tensorflow as tf
import numpy as np


class AndroidDataPreparer(DataPreparerDef):
    @staticmethod
    def default_settings():
        return {
            **DataPreparerDef.default_settings(),
            "connection": None,
            "guest_path": "/sdcard/accuracy_test"
        }

    def __init__(self, settings={}):
        super().__init__(settings)
        self.connection: Connection = self.settings["connection"]
        self.guest_path: str = self.settings["guest_path"]

    @staticmethod
    def _query_tflite_num_outputs(model_path):
        interpreter = tf.lite.Interpreter(model_path=model_path)
        output_details = interpreter.get_output_details()
        return np.prod(output_details[0]["shape"])

    def _prepare_models(self, model_paths):
        for model_path in model_paths:
            num_outputs = self._query_tflite_num_outputs(model_path)
            assert num_outputs == 1000 or num_outputs == 1001

            model_basename_noext = rm_ext(os.path.basename(model_path))
            model_output_labels = "{}_output_labels.txt".format(
                model_basename_noext)
            with open(model_output_labels, "w") as f:
                if num_outputs == 1000:
                    for i in range(1, 1001):
                        f.write("{}\n".format(i))
                else:
                    for i in range(1001):
                        f.write("{}\n".format(i))

            self.connection.push(model_output_labels, self.guest_path)
            self.connection.push(model_path, self.guest_path)

    def _prepare_dateset(self, image_id_range):
        validation_set_path = self.settings["validation_set_path"]

        with open("ground_truth_labels.txt", "w") as f:
            for i in image_id_range:
                f.write("{}\n".format(self.image_labels[i]))

        self.connection.shell(
            '; '.join([
                "if [ -e \"{ground_truth_images}\" ]",
                "then rm -r {ground_truth_images}",
                "fi",
                "mkdir {ground_truth_images}"
            ]).format(ground_truth_images="{}/{}".format(
                self.guest_path, "ground_truth_images")
            )
        )

        for i in image_id_range:
            self.connection.push(
                "{}/{}".format(validation_set_path, self.image_basenames[i]),
                "{}/{}".format(self.guest_path, "ground_truth_images")
            )

        self.connection.push("ground_truth_labels.txt", self.guest_path)
