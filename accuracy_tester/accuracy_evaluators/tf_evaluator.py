from .accuracy_evaluator_def import AccuracyEvaluatorDef
from .utils import count_dataset_size, construct_evaluating_progressbar, evaluate_outputs

from utils.tf_model_utils import load_graph, analyze_inputs_outputs

import tensorflow as tf
import itertools
import os
import numpy as np
import cv2


class TfEvaluator(AccuracyEvaluatorDef):
    def evaluate_models(self, model_paths, image_path_label_gen):
        model_tps = {}

        image_path_label_gen, dataset_size = \
            count_dataset_size(image_path_label_gen)

        for model_path in model_paths:
            image_path_label_gen, gen = itertools.tee(image_path_label_gen)

            model_basename = os.path.basename(model_path)
            model_tps[model_basename] = np.zeros((10,), dtype=np.int32)

            bar = construct_evaluating_progressbar(
                dataset_size, model_basename)
            bar.update(0)

            graph = load_graph(model_path)
            input_ops, output_ops = analyze_inputs_outputs(graph)
            assert len(input_ops) == 1 and len(output_ops) == 1

            with tf.Session(graph=graph) as sess:
                for i, (image_path, image_label) in enumerate(gen):
                    image = self.settings["preprocess"](image_path)
                    outputs = sess.run(
                        output_ops[0].outputs[0],
                        feed_dict={
                            input_ops[0].outputs[0]: image
                        }
                    )
                    model_tps[model_basename] += \
                        evaluate_outputs(
                            outputs[0], 10, self.settings["index_to_label"], image_label)
                    bar.update(i + 1)

            # progression bar ends
            print()

            print("[{}] current_accuracy = {}".format(
                model_basename,
                model_tps[model_basename] * 100.0 / dataset_size
            ))

        return model_tps
