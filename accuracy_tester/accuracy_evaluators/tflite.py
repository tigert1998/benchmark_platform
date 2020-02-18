import os
import datetime
import itertools

import numpy as np

from .accuracy_evaluator_def import AccuracyEvaluatorDef
from utils.utils import concatenate_flags, rm_ext
from utils.connection import Connection
from .utils import evaluate_outputs, count_dataset_size, construct_evaluating_progressbar

import tensorflow as tf
import cv2


class Tflite(AccuracyEvaluatorDef):
    @staticmethod
    def default_settings():
        return {
            **AccuracyEvaluatorDef.default_settings(),
            "connection": Connection(),

            # on guest
            "imagenet_accuracy_eval_path": None,
            "guest_path": "/sdcard/accuracy_test",
            "imagenet_accuracy_eval_flags": None,
        }

    def __init__(self, settings):
        super().__init__(settings)
        self.connection: Connection = self.settings["connection"]
        self.log_f = None

    def snapshot(self):
        res = super().snapshot()
        if isinstance(self.connection, Connection):
            dummys = [
                "imagenet_accuracy_eval_path",
                "guest_path", "imagenet_accuracy_eval_flags"
            ]
        else:
            dummys = []
        for item in dummys:
            res.pop(item)
        return res

    def _eval_on_guest(self, model_details, image_path_label_gen):
        guest_path = self.settings["guest_path"]

        ground_truth_images_path = \
            "{}/{}".format(guest_path, "ground_truth_images")

        model_tps = {}
        image_path_label_gen, dataset_size = count_dataset_size(
            image_path_label_gen)

        for model_basename in map(
                lambda model_detail: os.path.basename(model_detail.model_path),
                model_details):

            model_output_labels = "{}_output_labels.txt".format(
                rm_ext(model_basename))
            output_file_path = "{}/{}_{}".format(
                guest_path, rm_ext(model_basename), "output.csv")

            cmd = "{} {}".format(
                self.settings["imagenet_accuracy_eval_path"],
                concatenate_flags({
                    "model_file": "{}/{}".format(guest_path, model_basename),
                    "ground_truth_images_path": ground_truth_images_path,
                    "ground_truth_labels": "{}/{}".format(guest_path, "ground_truth_labels.txt"),
                    "model_output_labels": "{}/{}".format(guest_path, model_output_labels),
                    "output_file_path": output_file_path,
                    "num_images": 0,
                    **self.settings["imagenet_accuracy_eval_flags"]
                })
            )
            print(cmd)
            print(self.connection.shell(cmd))
            self.connection.pull(
                output_file_path,
                "."
            )

            with open(os.path.basename(output_file_path), "r") as f:
                for line in f:
                    pass
                accuracies = np.array(list(map(float, line.split(','))))
                print("[{}] current_accuracy = {}".format(
                    model_basename,
                    accuracies
                ))
                model_tps[model_basename] = np.round(
                    accuracies / 100. * dataset_size).astype(np.int32)

        return model_tps

    def _eval_on_host(self, model_details, image_path_label_gen):
        model_tps = {}

        image_path_label_gen, dataset_size = \
            count_dataset_size(image_path_label_gen)

        if self.log_f is None:
            self.log_f = open("log", "w")
        self.log_f.write(
            "len(image_path_label_gen) = {}\n".format(dataset_size))

        for model_detail in model_details:
            model_path = model_detail.model_path
            preprocess = model_detail.preprocess

            model_basename = os.path.basename(model_path)
            model_tps[model_basename] = np.zeros((10,), dtype=np.int32)

            image_path_label_gen, gen = itertools.tee(image_path_label_gen)

            interpreter = tf.lite.Interpreter(model_path=model_path)
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            assert len(input_details) == 1 and len(output_details) == 1

            bar = construct_evaluating_progressbar(
                dataset_size, model_basename)
            bar.update(0)

            for i, (image_path, image_label) in enumerate(gen):
                image = preprocess.execute(image_path)
                interpreter.set_tensor(input_details[0]["index"], image)
                interpreter.invoke()
                outputs = interpreter.get_tensor(output_details[0]["index"])
                model_tps[model_basename] += \
                    evaluate_outputs(outputs.flatten(), 10, image_label)

                bar.update(i + 1)

            print()
            self.log_f.write("time = {}\n".format(datetime.datetime.now()))
            self.log_f.write("model_tps[{}] = {}\n".format(
                model_basename,
                model_tps[model_basename]))
            self.log_f.flush()

        return model_tps

    def evaluate_models(self, model_details, image_path_label_gen):
        if type(self.connection) is Connection:
            return self._eval_on_host(model_details, image_path_label_gen)
        else:
            return self._eval_on_guest(model_details, image_path_label_gen)
