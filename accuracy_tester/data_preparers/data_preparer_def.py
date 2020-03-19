from utils.class_with_settings import ClassWithSettings
from glob import glob
import os


class DataPreparerDef(ClassWithSettings):
    @staticmethod
    def default_settings():
        return {
            **ClassWithSettings.default_settings(),
            "labels_path": None,
            "validation_set_path": None,
            "skip_models_preparation": False,
            "skip_dataset_preparation": False,
        }

    def __init__(self, settings={}):
        super().__init__(settings)
        self._init_label()

    def _init_label(self):
        self.image_basenames = []
        self.image_labels = []
        with open(self.settings["labels_path"]) as f:
            for line in f:
                image_basename, image_label =\
                    map(lambda x: x.strip(), line.split(' '))
                self.image_labels.append(image_label)
                self.image_basenames.append(image_basename)

    def image_path_label_gen(self, image_id_range):
        image_paths = glob(
            "{}/**/*".format(self.settings["validation_set_path"])
        )
        basename_to_path = dict()
        for image_path in image_paths:
            basename_to_path[os.path.basename(image_path)] = image_path

        for i in image_id_range:
            yield (
                basename_to_path[self.image_basenames[i]],
                self.image_labels[i]
            )

    def _prepare_models(self, model_paths):
        pass

    def prepare_models(self, model_paths):
        if self.settings["skip_models_preparation"]:
            return
        self._prepare_models(model_paths)

    def _prepare_dateset(self, image_id_range):
        pass

    def prepare_dateset(self, image_id_range):
        if self.settings["skip_dataset_preparation"]:
            return
        self._prepare_dateset(image_id_range)
