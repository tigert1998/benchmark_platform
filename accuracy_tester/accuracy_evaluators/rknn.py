from .accuracy_evaluator_def import AccuracyEvaluatorDef

from .utils import evaluate_outputs, count_dataset_size, construct_evaluating_progressbar

from rknn.api import RKNN

import cv2
import numpy as np
import itertools
import os

import progressbar


class Rknn(AccuracyEvaluatorDef):
    @staticmethod
    def default_settings():
        return {
            **AccuracyEvaluatorDef.default_settings(),
            "rknn_target": "rk1808",
        }

    def __init__(self, settings={}):
        super().__init__(settings)

    def brief(self):
        return "{}_{}".format(super().brief(), self.settings["rknn_target"])

    def evaluate_models(self, model_paths, image_path_label_gen):
        model_accuracies = {}

        image_path_label_gen, dataset_size = \
            count_dataset_size(image_path_label_gen)

        for model_path in model_paths:
            image_path_label_gen, gen = itertools.tee(image_path_label_gen)

            model_basename = os.path.basename(model_path)
            model_accuracies[model_basename] = np.zeros((10,))

            rknn = RKNN()

            assert 0 == rknn.load_rknn(model_path)

            assert 0 == rknn.init_runtime(target=self.settings["rknn_target"])

            bar = construct_evaluating_progressbar(
                dataset_size, model_basename)
            bar.update(0)

            for i, (image_path, image_label) in enumerate(gen):
                image = cv2.imread(image_path)[:, :, ::-1]
                image = self.settings["preprocess"](image)
                outputs = rknn.inference(inputs=[image])
                # assume that image_label is the index of output activation
                model_accuracies[model_basename] += \
                    evaluate_outputs(
                        outputs[0][0], 10, self.settings["index_to_label"], image_label)

                bar.update(i + 1)

            # progression bar ends
            print()

            model_accuracies[model_basename] = \
                model_accuracies[model_basename] * 100 / dataset_size
            print("[{}] current_accuracy = {}".format(
                model_basename, model_accuracies[model_basename]))

            rknn.release()

        return model_accuracies
