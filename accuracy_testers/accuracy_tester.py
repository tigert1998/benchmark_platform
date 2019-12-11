from utils.class_with_settings import ClassWithSettings
from utils.utils import adb_push, adb_shell, camel_case_to_snake_case, concatenate_flags, adb_pull

import zipfile
import os
import shutil

import numpy as np

import progressbar


class AccuracyTester(ClassWithSettings):
    @staticmethod
    def default_settings():
        return {
            "labels_path": None,
            "validation_images_path": None,
            "adb_device_id": None,
            "zip_size": 10,
        }

    def _clean_dir(self, dirname):
        if os.path.exists(dirname):
            shutil.rmtree(dirname)
        os.mkdir(dirname)

    def _read_labels(self):
        res = {}
        with open(self.settings["labels_path"]) as f:
            for line in f:
                filename, label = map(
                    lambda x: x.strip().lower(), line.split(' '))
                res[filename] = label
        return res

    def _get_dir_name(self):
        dir_name = "fk_" + \
            camel_case_to_snake_case(self.settings["class_name"])
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

    @staticmethod
    def _chdir_out():
        os.chdir("..")

    def run(self):
        self._chdir_in()

        with open("model_output_labels.txt", "w") as f:
            for i in range(1001):
                f.write("{}\n".format(i))

        image_to_label = self._read_labels()

        image_filenames = list(
            filter(
                lambda filename: filename.lower().endswith(".jpeg"),
                sorted(os.listdir(self.settings["validation_images_path"]))
            )
        )
        work_load = len(image_filenames)
        bar = progressbar.ProgressBar(
            max_value=work_load,
            redirect_stderr=False,
            redirect_stdout=False)
        bar.update(0)

        accuracies = np.zeros((10, ))

        for start in range(0, work_load, self.settings["zip_size"]):
            zip_size = min(work_load, start + self.settings["zip_size"]) - start

            print("zipping ...")
            with zipfile.ZipFile("ground_truth_images.zip", "w") as f:
                for i in range(start, start + zip_size):
                    f.write(
                        "{}/{}".format(
                            self.settings["validation_images_path"],
                            image_filenames[i]),
                        image_filenames[i]
                    )

            with open("ground_truth_labels.txt", "w") as f:
                for i in range(start, start + zip_size):
                    label = image_to_label[image_filenames[i].lower()]
                    f.write("{}\n".format(label))

            guest_path = "/sdcard/accuracy_test"
            adb_push(self.settings["adb_device_id"],
                     "ground_truth_images.zip", guest_path)
            adb_push(self.settings["adb_device_id"],
                     "ground_truth_labels.txt", guest_path)
            adb_push(self.settings["adb_device_id"],
                     "model_output_labels.txt", guest_path)

            adb_shell(
                self.settings["adb_device_id"],
                '; '.join([
                    "if [ -e \"{ground_truth_images}\" ]",
                    "then rm -r {ground_truth_images}",
                    "fi",
                    "mkdir {ground_truth_images} && unzip {ground_truth_images}.zip -d {ground_truth_images}"
                ]).format(ground_truth_images="{}/{}".format(
                    guest_path, "ground_truth_images")
                )
            )

            cmd = "{} {}".format(
                "/data/local/tmp/tf-r2.1-60afa4e/imagenet_accuracy_eval",
                concatenate_flags({
                    "model_file": "{}/{}".format(guest_path, "mobilenet_v2_1.0_224_frozen.tflite"),
                    "ground_truth_images_path": "{}/{}".format(guest_path, "ground_truth_images"),
                    "ground_truth_labels": "{}/{}".format(guest_path, "ground_truth_labels.txt"),
                    "model_output_labels": "{}/{}".format(guest_path, "model_output_labels.txt"),
                    "output_file_path": "{}/{}".format(guest_path, "output.csv"),
                    "num_images": 0,
                    "delegate": "gpu"
                })
            )
            print(cmd)
            print(adb_shell(self.settings["adb_device_id"], cmd))
            adb_pull(
                self.settings["adb_device_id"],
                "{}/{}".format(guest_path, "output.csv"), "."
            )

            with open("output.csv", "r") as f:
                for line in f:
                    pass
                tmp = np.array(list(map(float, line.split(','))))
                accuracies += tmp * zip_size
                print("current_accuracy = {}".format(tmp))
                print("accumulated_accuracy = {}".format(accuracies / (start + zip_size)))

            bar.update(start + zip_size)

        accuracies /= len(work_load)
        print("final_accuracy = {}".format(accuracies))

        self._chdir_out()
