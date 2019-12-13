from utils.class_with_settings import ClassWithSettings


class DataPreparerDef(ClassWithSettings):
    @staticmethod
    def default_settings():
        return {
            **ClassWithSettings.default_settings(),
            "labels_path": None,
            "image_id_to_path_func": None,
            "skip_models_preparation": False,
            "skip_dataset_preparation": False,
        }

    def __init__(self, settings={}):
        super().__init__(settings)
        self.image_to_label = self._image_to_label()

    def _image_to_label(self):
        res = {}
        with open(self.settings["labels_path"]) as f:
            for line in f:
                filename, label = map(
                    lambda x: x.strip().lower(), line.split(' '))
                res[filename] = label
        return res

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
