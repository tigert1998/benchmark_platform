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
            "model_paths": [],
            "zip_size": 5000,
            "dataset_size": 50000,
            "data_preparer": None,
            "accuracy_evaluator": None,
            "dirname": None
        }

    def snapshot(self):
        return {
            "time": "{}".format(datetime.now()),
            **super().snapshot(),
        }

    def _dump_snapshot(self):
        with open('snapshot.json', 'w') as f:
            f.write(json.dumps(regularize_for_json(self.snapshot()), indent=4))

    def _chdir_in(self):
        if self.settings.get("dirname") is not None:
            dir_name = self.settings["dirname"]
        else:
            dir_name = self.brief()
        if not os.path.isdir(dir_name):
            if os.path.exists(dir_name):
                print("os.path.exists(\"{}\")".format(dir_name))
                exit()
            else:
                os.mkdir(dir_name)
        os.chdir(dir_name)

    def _evaluate_models(self, model_paths):
        data_preparer = self.settings["data_preparer"]
        accuracy_evaluator = self.settings["accuracy_evaluator"]
        zip_size = self.settings["zip_size"]

        model_accuracies = {}
        for model_basename in map(os.path.basename, model_paths):
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
                model_paths,
                data_preparer.image_path_label_gen(image_id_range)
            )

            for model_basename in map(os.path.basename, model_paths):
                model_accuracies[model_basename] += tmp[model_basename] * num_images
                print("[{}] accumulated_accuracy = {}".format(
                    model_basename,
                    model_accuracies[model_basename] / (start + num_images))
                )

            bar.update(start + num_images)
            print()

        for model_basename in model_accuracies:
            model_accuracies[model_basename] /= dataset_size
        return model_accuracies

    @staticmethod
    def _chdir_out():
        os.chdir("..")

    def run(self):
        self._chdir_in()
        self._dump_snapshot()

        csv_writer = CSVWriter()
        titles = ["model_name"] + ["top {}".format(i + 1) for i in range(10)]

        model_accuracies = self._evaluate_models(self.settings["model_paths"])

        for model_basename, accuracies in model_accuracies.items():
            data = [model_basename] + list(map(str, list(accuracies)))
            csv_writer.update_data(
                "data.csv",
                titles=titles,
                data=data,
                is_resume=False,
            )

        self._chdir_out()
