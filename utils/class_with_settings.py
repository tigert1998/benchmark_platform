import warnings
import types
from .utils import camel_case_to_snake_case


class ClassWithSettings:
    def __init__(self, settings={}):
        self.settings = {
            "class_name": type(self).__name__,
            **self.default_settings()
        }
        self.update_settings(settings)

    def update_settings(self, settings):
        default_settings = self.default_settings()
        for key in default_settings:
            if key in settings:
                self.settings[key] = settings[key]
        self._check_settings_legality(settings)

    def _check_settings_legality(self, settings):
        default_settings = self.default_settings()
        for key in settings:
            if key not in default_settings:
                warnings.warn("invalid key in settings: {}".format(key))

    def snapshot(self):
        res = {}
        for key, value in self.settings.items():
            if issubclass(type(value), ClassWithSettings):
                res[key] = getattr(value, "snapshot")()
            else:
                res[key] = value
        return res

    def brief(self):
        res = camel_case_to_snake_case(type(self).__name__)
        for _, value in self.settings.items():
            if issubclass(type(value), ClassWithSettings):
                res += "_" + getattr(value, "brief")()
        return res

    @staticmethod
    def default_settings():
        return {}
