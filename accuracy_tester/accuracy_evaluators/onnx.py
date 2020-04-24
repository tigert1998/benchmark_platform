from .accuracy_evaluator_def import AccuracyEvaluatorDef
from .utils import count_dataset_size, construct_evaluating_progressbar, evaluate_outputs

import itertools
import os
import numpy as np

import onnx
import onnxruntime as rt


class Onnx(AccuracyEvaluatorDef):
    def evaluate_models(self, model_details, image_path_label_gen):
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

            sess = rt.InferenceSession(model_path)
            input_node = sess.get_inputs()[0].name
            output_node = sess.get_outputs()[0].name

            input_shape = sess.get_inputs()[0].shape
            if input_shape[-1] == 3:
                is_bhwc = True
            elif input_shape[1] == 3:
                is_bhwc = False
            else:
                assert False

            for i, (image_path, image_label) in enumerate(gen):
                image = preprocess.execute(image_path)
                if not is_bhwc and image.shape[-1] == 3:
                    image = image.transpose((0, 3, 1, 2))
                elif is_bhwc and image.shape[1] == 3:
                    image = image.transpose((0, 2, 3, 1))

                outputs = sess.run(
                    [output_node], {
                        input_node: image
                    }
                )
                outputs = np.array(outputs)
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
