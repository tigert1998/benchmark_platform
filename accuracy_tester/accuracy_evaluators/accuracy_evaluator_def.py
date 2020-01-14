from utils.class_with_settings import ClassWithSettings


class AccuracyEvaluatorDef(ClassWithSettings):
    def __init__(self, settings={}):
        super().__init__(settings)

    @staticmethod
    def default_settings():
        return {
            **ClassWithSettings.default_settings(),
            "preprocess": lambda image: image,
            "index_to_label": lambda index: str(index)
        }

    def evaluate_models(self, model_paths, image_path_label_gen):
        """evaluate models' accuracies

        Args:
            model_paths: model paths in host machine
            image_path_label_gen: A generator which generates (image_path, image_label)

        Returns:
            Dict[model_basename, TPs]
        """
        pass
