from .data_preparer_def import DataPreparerDef
from utils.utils import adb_push, adb_shell

import os


class AndroidDataPreparer(DataPreparerDef):
    @staticmethod
    def default_settings():
        return {
            **DataPreparerDef.default_settings(),
            "adb_device_id": None,
            "guest_path": "/sdcard/accuracy_test"
        }

    def _prepare_models(self, model_paths):
        for model_path in model_paths:
            adb_push(
                self.settings["adb_device_id"],
                model_path, self.settings["guest_path"]
            )

    def _prepare_dateset(self, image_id_range):
        guest_path = self.settings["guest_path"]
        adb_device_id = self.settings["adb_device_id"]
        image_id_to_path_func = self.settings["image_id_to_path_func"]

        with open("ground_truth_labels.txt", "w") as f:
            for basename in map(os.path.basename, map(image_id_to_path_func, image_id_range)):
                label = self.image_to_label[basename.lower()]
                f.write("{}\n".format(label))

        with open("model_output_labels.txt", "w") as f:
            for i in range(1001):
                f.write("{}\n".format(i))

        adb_shell(
            adb_device_id,
            '; '.join([
                "if [ -e \"{ground_truth_images}\" ]",
                "then rm -r {ground_truth_images}",
                "fi",
                "mkdir {ground_truth_images}"
            ]).format(ground_truth_images="{}/{}".format(
                guest_path, "ground_truth_images")
            )
        )

        for i in image_id_range:
            adb_push(
                adb_device_id,
                image_id_to_path_func(i),
                "{}/{}".format(guest_path, "ground_truth_images")
            )

        adb_push(adb_device_id, "ground_truth_labels.txt", guest_path)
        adb_push(adb_device_id, "model_output_labels.txt", guest_path)
