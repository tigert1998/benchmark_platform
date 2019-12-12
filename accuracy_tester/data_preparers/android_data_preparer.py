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

        print("zipping ...")

        with tarfile.open("ground_truth_images.tar", "w:") as f:
            for basename in image_basenames:
                f.add(
                    "{}/{}".format(
                        self.settings["validation_images_path"],
                        basename),
                    basename
                )

        with open("ground_truth_labels.txt", "w") as f:
            for basename in image_basenames:
                label = self.image_to_label[basename.lower()]
                f.write("{}\n".format(label))

        adb_push(adb_device_id, "ground_truth_images.tar", guest_path)
        adb_push(adb_device_id, "ground_truth_labels.txt", guest_path)
        adb_push(adb_device_id, "model_output_labels.txt", guest_path)

        print("unzipping ...")

        adb_shell(
            adb_device_id,
            '; '.join([
                "if [ -e \"{ground_truth_images}\" ]",
                "then rm -r {ground_truth_images}",
                "fi",
                "mkdir {ground_truth_images} && tar -xvf {ground_truth_images}.tar -C {ground_truth_images}"
            ]).format(ground_truth_images="{}/{}".format(
                guest_path, "ground_truth_images")
            )
        )
