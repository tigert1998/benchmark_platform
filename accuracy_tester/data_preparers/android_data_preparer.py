from .data_preparer_def import DataPreparerDef
from utils.utils import adb_push, adb_shell

import os

import tensorflow as tf
import numpy as np


class AndroidDataPreparer(DataPreparerDef):
    @staticmethod
    def default_settings():
        return {
            **DataPreparerDef.default_settings(),
            "adb_device_id": None,
            "guest_path": "/sdcard/accuracy_test"
        }

    @staticmethod
    def _query_tflite_num_outputs(model_path):
        interpreter = tf.lite.Interpreter(model_path=model_path)
        output_details = interpreter.get_output_details()
        return np.prod(output_details[0]["shape"])

    def _prepare_models(self, model_paths):
        adb_device_id = self.settings["adb_device_id"]
        guest_path = self.settings["guest_path"]

        for model_path in model_paths:
            num_outputs = self._query_tflite_num_outputs(model_path)
            assert num_outputs == 1000 or num_outputs == 1001

            model_basename_noext = ".".join(
                os.path.basename(model_path).split(".")[:-1])
            model_output_labels = "{}_output_labels.txt".format(
                model_basename_noext)
            with open(model_output_labels, "w") as f:
                if num_outputs == 1000:
                    for i in range(1, 1001):
                        f.write("{}\n".format(i))
                else:
                    for i in range(1001):
                        f.write("{}\n".format(i))

            adb_push(adb_device_id, model_output_labels, guest_path)
            adb_push(adb_device_id, model_path, guest_path)

    def _prepare_dateset(self, image_id_range):
        guest_path = self.settings["guest_path"]
        adb_device_id = self.settings["adb_device_id"]
        validation_set_path = self.settings["validation_set_path"]

        with open("ground_truth_labels.txt", "w") as f:
            for i in image_id_range:
                f.write("{}\n".format(self.image_labels[i]))

        adb_shell(
            adb_device_id,
            '; '.join([
                "if [ -e \"{ground_truth_images}\" ]",
                "then rm -r {ground_truth_images}",
                "fi",
                "mkdir {ground_truth_images}"
            ]).format(ground_truth_images="{}/{}".format(
                guest_path, "ground_truth_images")
            )
        )

        for i in image_id_range:
            adb_push(
                adb_device_id,
                "{}/{}".format(validation_set_path, self.image_basenames[i]),
                "{}/{}".format(guest_path, "ground_truth_images")
            )

        adb_push(adb_device_id, "ground_truth_labels.txt", guest_path)
