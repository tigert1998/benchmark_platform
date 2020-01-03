from utils.class_with_settings import ClassWithSettings


class Sampler(ClassWithSettings):
    @staticmethod
    def default_settings():
        return {
            **ClassWithSettings.default_settings(),
            "filter": lambda x: True,
            "channel_step": 4,
        }

    @staticmethod
    def get_sample_titles():
        """get sample points' titles
        Returns: List of titles
        """
        pass

    def _get_samples_without_filter(self):
        pass

    def get_samples(self):
        """get testing samples for a certain operator
        Returns: A generator of sample lists
        """
        return filter(self.settings["filter"], self._get_samples_without_filter())
