class ClassWithSettings:
    def __init__(self, settings={}):
        self.settings = {
            "class_name": type(self).__name__,
        }
        default_settings = self.default_settings()
        for key in default_settings:
            if key in settings:
                self.settings[key] = settings[key]
            else:
                self.settings[key] = default_settings[key]

    @staticmethod
    def default_settings():
        return {}
