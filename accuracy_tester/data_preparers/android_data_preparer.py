from .data_preparer_def import DataPreparerDef
from utils.utils import adb_push, adb_shell

import tarfile


class AndroidDataPreparer(DataPreparerDef):
    @staticmethod
    def default_settings():
        return {
            **DataPreparerDef.default_settings(),
            "adb_device_id": None,
            "guest_path": None
        }

    def _prepare(self, image_basenames):
        guest_path = self.settings["guest_path"]
        adb_device_id = self.settings["adb_device_id"]

        with open("ground_truth_labels.txt", "w") as f:
            for basename in image_basenames:
                label = self.image_to_label[basename.lower()]
                f.write("{}\n".format(label))

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

        for basename in sorted(image_basenames):
            adb_push(
                adb_device_id,
                "{}/{}".format(
                    self.settings["validation_images_path"],
                    basename),
                "{}/{}".format(guest_path, "ground_truth_images")
            )

        adb_push(adb_device_id, "ground_truth_labels.txt", guest_path)
        adb_push(adb_device_id, "model_output_labels.txt", guest_path)
