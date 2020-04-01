from .accuracy_evaluator_def import AccuracyEvaluatorDef
from .utils import count_dataset_size, construct_evaluating_progressbar, evaluate_outputs

import tensorflow as tf
import itertools
import os
import numpy as np


class Tpu(AccuracyEvaluatorDef):
    @staticmethod
    def default_settings():
        return {
            **AccuracyEvaluatorDef.default_settings(),
            "libedgetpu_path": "libedgetpu.so.1"
        }

    def __init__(self, settings={}):
        super().__init__(settings)

        import tflite_runtime.interpreter as tflite
        self.delegate = tflite.load_delegate(self.settings["libedgetpu_path"])

    def evaluate_models(self, model_details, image_path_label_gen):
        import tflite_runtime.interpreter as tflite

        model_tps = {}

        image_path_label_gen, dataset_size = \
            count_dataset_size(image_path_label_gen)

        for model_detail in model_details:
            model_path = model_detail.model_path
            preprocess = model_detail.preprocess

            image_path_label_gen, gen = itertools.tee(image_path_label_gen)

            model_basename = os.path.basename(model_path)
            model_tps[model_basename] = np.zeros((10,), dtype=np.int32)

            bar = construct_evaluating_progressbar(
                dataset_size, model_basename)
            bar.update(0)

            interpreter = tflite.Interpreter(
                model_path=model_path,
                experimental_delegates=[self.delegate]
            )
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            interpreter.allocate_tensors()
            for i, (image_path, image_label) in enumerate(gen):
                image = preprocess.execute(image_path)
                interpreter.set_tensor(
                    input_details[0]["index"],
                    image
                )
                interpreter.invoke()
                outputs = interpreter.get_tensor(output_details[0]["index"])

                model_tps[model_basename] += \
                    evaluate_outputs(
                    outputs.flatten(), 10, image_label)
                bar.update(i + 1)

            # progression bar ends
            print()

            print("[{}] current_accuracy = {}".format(
                model_basename,
                model_tps[model_basename] * 100.0 / dataset_size
            ))

        return model_tps
