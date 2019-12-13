from accuracy_tester.accuracy_tester import AccuracyTester
from accuracy_tester.data_preparers.data_preparer_def import DataPreparerDef
from accuracy_tester.accuracy_evaluators.rknn import Rknn

import glob


def main():
    tester = AccuracyTester({
        "model_paths": [],
        "data_preparer": DataPreparerDef({
            "labels_path": "C:/Users/v-xiat/Downloads/imagenet/val_labels.txt",
            "validation_set_path": "C:/Users/v-xiat/Downloads/imagenet/validation",
            "skip_models_preparation": True,
            "skip_dataset_preparation": True,
        }),
        "accuracy_evaluator": Rknn({
        })
    })
    tester.run()


if __name__ == '__main__':
    main()
