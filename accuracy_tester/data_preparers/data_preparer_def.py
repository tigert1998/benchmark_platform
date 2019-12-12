from utils.class_with_settings import ClassWithSettings


class DataPreparerDef(ClassWithSettings):
    @staticmethod
    def default_settings():
        return {
            **ClassWithSettings.default_settings(),
            "labels_path": None,
            "validation_images_path": None,
            "skip": False
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

    def _prepare(self, image_basenames):
        pass

    def prepare(self, image_basenames):
        if self.settings["skip"]:
            return
        self._prepare(image_basenames)
