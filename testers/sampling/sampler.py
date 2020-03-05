from utils.class_with_settings import ClassWithSettings


class Sampler(ClassWithSettings):
    @staticmethod
    def default_settings():
        return {
            **ClassWithSettings.default_settings(),
            "filter": lambda x: True,
        }

    @staticmethod
    def get_sample_titles():
        """Get sample' titles. This method should not be called outside the class.
        Returns: List of titles
        """
        pass

    @staticmethod
    def _get_serializable_sample(sample):
        """Make sample writable. This method should not be called outside the class.
        """
        return sample

    def get_sample_dict(self, sample):
        """Call self.get_sample_titles and self._get_serializable_sample and concatenate results to a dict.
        """
        return {
            key: value for key, value in zip(
                self.get_sample_titles(),
                self._get_serializable_sample(sample)
            )
        }

    def _get_samples_without_filter(self):
        pass

    def get_samples(self):
        """get testing samples for a certain operator
        Returns: A generator of sample lists
        """
        return filter(self.settings["filter"], self._get_samples_without_filter())
