from .accuracy_evaluator_def import AccuracyEvaluatorDef

from utils.std_preprocess import std_preprocess_single_image

from rknn.api import RKNN

import cv2
import numpy as np
import itertools
import os

import progressbar


def evaluate_outputs(activations, top_n, answer):
    assert activations.shape == (1001,)
    indices = map(
        lambda pair: pair[0],
        itertools.islice(
            sorted(
                enumerate(activations),
                key=lambda pair: -pair[1]
            ),
            top_n
        )
    )
    ret = []
    current = False
    for index in indices:
        current = current or (index == answer)
        ret.append(int(current))
    return ret


class Rknn(AccuracyEvaluatorDef):
    @staticmethod
    def default_settings():
        return {
            **AccuracyEvaluatorDef.default_settings(),
            "rknn_target": "rk1808"
        }

    def brief(self):
        return "{}_{}".format(super().brief(), self.settings["rknn_target"])

    @staticmethod
    def count_dataset_size(image_path_label_gen):
        """count_dataset_size
        Returns:
            (generator, int)
        """
        image_path_label_gen, gen = itertools.tee(image_path_label_gen)
        ret = 0
        for _ in gen:
            ret += 1
        return (image_path_label_gen, ret)

    def evaluate_models(self, model_paths, image_path_label_gen):
        model_accuracies = {}

        image_path_label_gen, dataset_size = \
            self.count_dataset_size(image_path_label_gen)

        for model_path in model_paths:
            image_path_label_gen, gen = itertools.tee(image_path_label_gen)

            model_basename = os.path.basename(model_path)
            model_accuracies[model_basename] = np.zeros((10,))

            rknn = RKNN()

            assert 0 == rknn.load_rknn(model_path)

            assert 0 == rknn.init_runtime(target=self.settings["rknn_target"])

            bar = progressbar.ProgressBar(max_value=dataset_size, widgets=[
                'Evaluating {} ['.format(model_basename),
                progressbar.Timer(),
                '] ',
                progressbar.Bar(),
                ' (',
                progressbar.ETA(),
                ') ',
            ])
            bar.update(0)

            for i, (image_path, image_label) in enumerate(gen):
                image = cv2.imread(image_path)[:, :, ::-1]
                image = std_preprocess_single_image(image, 224, np.uint8)
                outputs = rknn.inference(inputs=[image])
                # assume that image_label is the index of output activation
                model_accuracies[model_basename] += \
                    evaluate_outputs(outputs[0][0], 10, int(image_label))

                bar.update(i + 1)

            # progression bar ends
            print()

            model_accuracies[model_basename] = \
                model_accuracies[model_basename] * 100 / dataset_size
            print("[{}] current_accuracy = {}".format(
                model_basename, model_accuracies[model_basename]))

            rknn.release()

        return model_accuracies
