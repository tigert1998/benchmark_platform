import os
import numpy as np
import itertools

from .accuracy_evaluator_def import AccuracyEvaluatorDef
from utils.utils import \
    adb_shell, adb_push, adb_pull, \
    concatenate_flags, inquire_adb_device
from .utils import evaluate_outputs, count_dataset_size, construct_evaluating_progressbar

import tensorflow as tf
import cv2


class Tflite(AccuracyEvaluatorDef):
    @staticmethod
    def default_settings():
        return {
            **AccuracyEvaluatorDef.default_settings(),
            "eval_on_host": False,

            # on guest
            "adb_device_id": None,
            "imagenet_accuracy_eval_path": None,
            "guest_path": "/sdcard/accuracy_test",
            "imagenet_accuracy_eval_flags": None,

            # on host
            "preprocess": lambda image: image,
            "index_to_label": lambda index: str(index)
        }

    def snapshot(self):
        res = super().snapshot()
        if self.settings["eval_on_host"]:
            dummys = [
                "adb_device_id", "imagenet_accuracy_eval_path",
                "guest_path", "imagenet_accuracy_eval_flags"
            ]
        else:
            dummys = ["preprocess", "index_to_label"]
            res["adb_device_id"] =\
                inquire_adb_device(self.settings["adb_device_id"])
        for item in dummys:
            res.pop(item)
        return res

    def brief(self):
        device_info = "host" if self.settings["eval_on_host"] else self.settings["adb_device_id"]
        return "{}_{}".format(super().brief(), device_info)

    def _eval_on_guest(self, model_paths, image_path_label_gen):
        guest_path = self.settings["guest_path"]
        adb_device_id = self.settings["adb_device_id"]

        ground_truth_images_path = \
            "{}/{}".format(guest_path, "ground_truth_images")

        model_accuracies = {}

        for model_basename in map(os.path.basename, model_paths):
            model_basename_noext = ".".join(model_basename.split(".")[:-1])
            model_output_labels = "{}_output_labels.txt".format(
                model_basename_noext)

            cmd = "{} {}".format(
                self.settings["imagenet_accuracy_eval_path"],
                concatenate_flags({
                    "model_file": "{}/{}".format(guest_path, model_basename),
                    "ground_truth_images_path": ground_truth_images_path,
                    "ground_truth_labels": "{}/{}".format(guest_path, "ground_truth_labels.txt"),
                    "model_output_labels": "{}/{}".format(guest_path, model_output_labels),
                    "output_file_path": "{}/{}".format(guest_path, "output.csv"),
                    "num_images": 0,
                    **self.settings["imagenet_accuracy_eval_flags"]
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

    def _eval_on_host(self, model_paths, image_path_label_gen):
        model_accuracies = {}

        image_path_label_gen, dataset_size = \
            count_dataset_size(image_path_label_gen)

        for model_path in model_paths:
            model_basename = os.path.basename(model_path)
            model_accuracies[model_basename] = np.zeros((10,))

            image_path_label_gen, gen = itertools.tee(image_path_label_gen)

            interpreter = tf.lite.Interpreter(model_path=model_path)
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            assert len(input_details) == 1 and len(output_details) == 1

            bar = construct_evaluating_progressbar(
                dataset_size, model_basename)
            bar.update(0)

            for i, (image_path, image_label)in enumerate(gen):
                image = cv2.imread(image_path)[:, :, ::-1]
                image = self.settings["preprocess"](image)
                interpreter.set_tensor(input_details[0]["index"], image)
                interpreter.invoke()
                outputs = interpreter.get_tensor(output_details[0]["index"])
                model_accuracies[model_basename] += \
                    evaluate_outputs(
                        outputs[0], 10,
                        self.settings["index_to_label"],
                    image_label
                )

                bar.update(i + 1)

            model_accuracies[model_basename] *= 100 / dataset_size

        return model_accuracies

    def evaluate_models(self, model_paths, image_path_label_gen):
        if self.settings["eval_on_host"]:
            return self._eval_on_host(model_paths, image_path_label_gen)
        else:
            return self._eval_on_guest(model_paths, image_path_label_gen)
