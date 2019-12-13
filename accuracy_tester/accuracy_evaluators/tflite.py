import os
import numpy as np

from .accuracy_evaluator_def import AccuracyEvaluatorDef
from utils.utils import \
    adb_shell, adb_push, adb_pull, \
    concatenate_flags, inquire_adb_device


class Tflite(AccuracyEvaluatorDef):
    @staticmethod
    def default_settings():
        return {
            **AccuracyEvaluatorDef.default_settings(),
            "adb_device_id": None,
            "imagenet_accuracy_eval_path": None,
            "guest_path": "/sdcard/accuracy_test",
            "delegate": ""
        }

    def snapshot(self):
        res = super().snapshot()
        res["adb_device_id"] = inquire_adb_device(
            self.settings["adb_device_id"])
        return res

    def brief(self):
        return "{}_{}".format(super().brief(), self.settings["adb_device_id"])

    def evaluate_models(self, model_paths, image_path_label_gen):
        guest_path = self.settings["guest_path"]
        adb_device_id = self.settings["adb_device_id"]

        ground_truth_images_path = \
            "{}/{}".format(guest_path, "ground_truth_images")

        model_accuracies = {}

        for model_basename in map(os.path.basename, model_paths):
            cmd = "{} {}".format(
                self.settings["imagenet_accuracy_eval_path"],
                concatenate_flags({
                    "model_file": "{}/{}".format(guest_path, model_basename),
                    "ground_truth_images_path": ground_truth_images_path,
                    "ground_truth_labels": "{}/{}".format(guest_path, "ground_truth_labels.txt"),
                    "model_output_labels": "{}/{}".format(guest_path, "model_output_labels.txt"),
                    "output_file_path": "{}/{}".format(guest_path, "output.csv"),
                    "num_images": 0,
                    "delegate": self.settings["delegate"]
                })
            )
            print(cmd)
            print(adb_shell(adb_device_id, cmd))
            adb_pull(
                adb_device_id,
                "{}/{}".format(guest_path, "output.csv"),
                "."
            )

            with open("output.csv", "r") as f:
                for line in f:
                    pass
                model_accuracies[model_basename] =\
                    np.array(list(map(float, line.split(','))))
                print("[{}] current_accuracy = {}".format(
                    model_basename,
                    model_accuracies[model_basename])
                )

        return model_accuracies
