from utils.class_with_settings import ClassWithSettings


class AccuracyEvaluatorDef(ClassWithSettings):
    def __init__(self, settings={}):
        super().__init__(settings)

    @staticmethod
    def default_settings():
        return {
            **ClassWithSettings.default_settings(),
        }

    def evaluate_models(self, model_paths):
        """evaluate models' accuracies
        Args:
            model_paths: model paths in host machine
        Returns:
            Dict[ModelBaseName, Accuracies]
        """
        pass
