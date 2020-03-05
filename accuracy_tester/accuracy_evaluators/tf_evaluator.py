from .accuracy_evaluator_def import AccuracyEvaluatorDef
from .utils import count_dataset_size, construct_evaluating_progressbar, evaluate_outputs

from utils.tf_model_utils import load_graph

import tensorflow as tf
import itertools
import os
import numpy as np


class TfEvaluator(AccuracyEvaluatorDef):
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

            graph = load_graph(model_path)

            with tf.Session(graph=graph) as sess:
                for i, (image_path, image_label) in enumerate(gen):
                    image = preprocess.execute(image_path)
                    outputs = sess.run(
                        model_detail.output_node + ":0",
                        feed_dict={
                            model_detail.input_node + ":0":
                            image
                        }
                    )
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
