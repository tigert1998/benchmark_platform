from utils.class_with_settings import ClassWithSettings


class AccuracyEvaluatorDef(ClassWithSettings):
    def evaluate_models(self, model_details, image_path_label_gen):
        """evaluate models' accuracies

        Args:
            model_details: [ModelDetail]
            image_path_label_gen: A generator which generates (image_path, image_label)

        Returns:
            Dict[model_basename, TPs]
        """
        pass
