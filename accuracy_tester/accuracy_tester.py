from utils.class_with_settings import ClassWithSettings
from utils.utils import \
    camel_case_to_snake_case, regularize_for_json,\
    concatenate_flags
from utils.connection import Connection

from .data_preparers.android_data_preparer import AndroidDataPreparer
from utils.csv_writer import CSVWriter

import os
import json
import numpy as np
from datetime import datetime

import progressbar


class AccuracyTester(ClassWithSettings):
    @staticmethod
    def default_settings():
        return {
            **ClassWithSettings.default_settings(),
            "model_details": [],
            "zip_size": 5000,
            "dataset_size": 50000,
            "data_preparer": None,
            "accuracy_evaluator": None,
            "dirname": None
        }

    def snapshot(self):
        snapshot = super().snapshot()
        snapshot["time"] = "{}".format(datetime.now())
        snapshot["model_details"] = []
        for model_detail in self.settings["model_details"]:
            snapshot["model_details"].append([
                model_detail.model_path,
                model_detail.preprocess.snapshot()
            ])
        return snapshot

    def _dump_snapshot(self):
        with open('snapshot.json', 'w') as f:
            f.write(json.dumps(regularize_for_json(self.snapshot()), indent=4))

    def _chdir_in(self):
        self.cwd = os.getcwd()
        if self.settings.get("dirname") is not None:
            dir_name = self.settings["dirname"]
        else:
            dir_name = self.brief()
        dir_name = "test_results/{}".format(dir_name)
        if not os.path.isdir(dir_name):
            if os.path.exists(dir_name):
                print("os.path.exists(\"{}\")".format(dir_name))
                exit()
            else:
                os.makedirs(dir_name)
        os.chdir(dir_name)

    def _evaluate_models(self, model_details):
        data_preparer = self.settings["data_preparer"]
        accuracy_evaluator = self.settings["accuracy_evaluator"]
        zip_size = self.settings["zip_size"]

        model_paths = list(map(
            lambda model_detail: model_detail.model_path,
            model_details))
        model_basenames = list(map(os.path.basename, model_paths))

        model_accuracies = {}
        for model_basename in model_basenames:
            model_accuracies[model_basename] = np.zeros((10, ))

        dataset_size = self.settings["dataset_size"]
        bar = progressbar.ProgressBar(
            max_value=dataset_size,
            redirect_stderr=False,
            redirect_stdout=False)
        bar.update(0)
        print()

        data_preparer.prepare_models(model_paths)

        for start in range(0, dataset_size, zip_size):
            num_images = min(dataset_size, start + zip_size) - start

            image_id_range = range(start, start + num_images)
            data_preparer.prepare_dateset(image_id_range)

            tmp = accuracy_evaluator.evaluate_models(
                model_details,
                data_preparer.image_path_label_gen(image_id_range)
            )

            for model_basename in model_basenames:
                model_accuracies[model_basename] += tmp[model_basename]
                print("[{}] accumulated_accuracy = {}".format(
                    model_basename,
                    100.0 * model_accuracies[model_basename] / (start + num_images))
                )

            bar.update(start + num_images)
            print()

        for model_basename in model_accuracies:
            model_accuracies[model_basename] *= 100 / dataset_size
        return model_accuracies

    def _chdir_out(self):
        os.chdir(self.cwd)

    def run(self):
        self._chdir_in()
        self._dump_snapshot()

        csv_writer = CSVWriter()
        titles = ["model_name"] + ["top {}".format(i + 1) for i in range(10)]

        model_accuracies = self._evaluate_models(
            self.settings["model_details"])

        for model_basename, accuracies in model_accuracies.items():
            data = [model_basename] + list(map(str, list(accuracies)))
            csv_writer.update_data(
                "data.csv",
                data=dict(zip(titles, data)),
                is_resume=False,
            )

        self._chdir_out()
