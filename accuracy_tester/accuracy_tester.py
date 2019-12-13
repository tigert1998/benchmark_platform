from utils.class_with_settings import ClassWithSettings
from utils.utils import \
    adb_shell, adb_push, adb_pull, \
    camel_case_to_snake_case, concatenate_flags

from .data_preparers.android_data_preparer import AndroidDataPreparer
from utils.csv_writer import CSVWriter

import os
import shutil

import numpy as np

import progressbar


class AccuracyTester(ClassWithSettings):
    @staticmethod
    def default_settings():
        return {
            **ClassWithSettings.default_settings(),
            "labels_path": None,
            "validation_images_path": None,
            "model_paths": [],
            "adb_device_id": None,
            "skip_data_preparation": False,
            "zip_size": 5000,
            "ILSVRC2012_val_size": 50000,
        }

    def _clean_dir(self, dirname):
        if os.path.exists(dirname):
            shutil.rmtree(dirname)
        os.mkdir(dirname)

    def _get_dir_name(self):
        dir_name = "{}_{}".format(
            camel_case_to_snake_case(self.settings["class_name"]),
            self.settings["adb_device_id"]
        )
        return dir_name

    def _chdir_in(self):
        dir_name = self._get_dir_name()
        if not os.path.isdir(dir_name):
            if os.path.exists(dir_name):
                print("os.path.exists(\"{}\")".format(dir_name))
                exit()
            else:
                os.mkdir(dir_name)
        os.chdir(dir_name)

    def _evaluate_models(self, model_paths, guest_path):
        model_accuracies = {}
        for model_path in model_paths:
            adb_push(self.settings["adb_device_id"], model_path, guest_path)
            model_accuracies[os.path.basename(model_path)] = np.zeros((10, ))

        image_basenames = list(
            filter(
                lambda filename: filename.lower().endswith(".jpeg"),
                sorted(os.listdir(self.settings["validation_images_path"]))
            )
        )
        work_load = len(image_basenames)
        bar = progressbar.ProgressBar(
            max_value=work_load,
            redirect_stderr=False,
            redirect_stdout=False)
        bar.update(0)
        print()

        data_preparer = AndroidDataPreparer({
            "labels_path": self.settings["labels_path"],
            "validation_images_path": self.settings["validation_images_path"],
            "skip": self.settings["skip_data_preparation"],
            "adb_device_id": self.settings["adb_device_id"],
            "guest_path": guest_path
        })

        for start in range(0, work_load, self.settings["zip_size"]):
            zip_size = min(work_load, start +
                           self.settings["zip_size"]) - start

            data_preparer.prepare(image_basenames[start: start + zip_size])

            # evaluate a single model
            for model_path in model_paths:
                model_basename = os.path.basename(model_path)
                cmd = "{} {}".format(
                    "/data/local/tmp/tf-r2.1-60afa4e/imagenet_accuracy_eval",
                    concatenate_flags({
                        "model_file": "{}/{}".format(guest_path, model_basename),
                        "ground_truth_images_path": "{}/{}".format(guest_path, "ground_truth_images"),
                        "ground_truth_labels": "{}/{}".format(guest_path, "ground_truth_labels.txt"),
                        "model_output_labels": "{}/{}".format(guest_path, "model_output_labels.txt"),
                        "output_file_path": "{}/{}".format(guest_path, "output.csv"),
                        "num_images": 0,
                        "delegate": ""
                    })
                )
                print(cmd)
                print(adb_shell(self.settings["adb_device_id"], cmd))
                adb_pull(
                    self.settings["adb_device_id"],
                    "{}/{}".format(guest_path, "output.csv"),
                    "."
                )

                with open("output.csv", "r") as f:
                    for line in f:
                        pass
                    tmp = np.array(list(map(float, line.split(','))))
                    model_accuracies[model_basename] += tmp * zip_size
                    print("[{}] current_accuracy = {}".format(
                        model_basename,
                        tmp)
                    )
                    print("[{}] accumulated_accuracy = {}".format(
                        model_basename,
                        model_accuracies[model_basename] / (start + zip_size))
                    )

            bar.update(start + zip_size)
            print()

        for model_basename in model_accuracies:
            model_accuracies[model_basename] /= work_load
        return model_accuracies

    @staticmethod
    def _chdir_out():
        os.chdir("..")

    def run(self):
        self._chdir_in()

        with open("model_output_labels.txt", "w") as f:
            for i in range(1001):
                f.write("{}\n".format(i))

        csv_writer = CSVWriter()
        titles = ["model_name"] + ["top {}".format(i + 1) for i in range(10)]

        guest_path = "/sdcard/accuracy_test"
        model_accuracies = self._evaluate_models(
            self.settings["model_paths"],
            guest_path)

        for model_basename, accuracies in model_accuracies:
            data = [model_basename] + list(map(str, list(accuracies)))
            csv_writer.update_data(
                "data.csv",
                titles=titles,
                data=data,
                is_resume=False,
            )

        self._chdir_out()
