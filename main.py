from glob import glob


def accuracy_test_rknn():
    from accuracy_tester.accuracy_tester import AccuracyTester
    from accuracy_tester.data_preparers.data_preparer_def import DataPreparerDef
    from accuracy_tester.accuracy_evaluators.rknn import Rknn

    tester = AccuracyTester({
        "zip_size": 50000,
        "model_paths": glob("C:/Users/v-xiat/Downloads/imagenet/models/*.rknn"),
        "data_preparer": DataPreparerDef({
            "labels_path": "C:/Users/v-xiat/Downloads/imagenet/val_labels.txt",
            "validation_set_path": "C:/Users/v-xiat/Downloads/imagenet/validation",
            "skip_models_preparation": True,
            "skip_dataset_preparation": True,
        }),
        "accuracy_evaluator": Rknn({})
    })
    tester.run()


def accuracy_test_tflite():
    from accuracy_tester.accuracy_tester import AccuracyTester
    from accuracy_tester.data_preparers.android_data_preparer import AndroidDataPreparer
    from accuracy_tester.accuracy_evaluators.tflite import Tflite

    tester = AccuracyTester({
        "zip_size": 50000,
        "model_paths": [
            "C:/Users/v-xiat/Downloads/imagenet/models/mnasnet-a1_float16_quant.tflite",
        ],
        "data_preparer": AndroidDataPreparer({
            "labels_path": "C:/Users/v-xiat/Downloads/imagenet/val_labels.txt",
            "validation_set_path": "C:/Users/v-xiat/Downloads/imagenet/validation",
            "skip_models_preparation": False,
            "skip_dataset_preparation": True,
            "adb_device_id": "5e6fecf"
        }),
        "accuracy_evaluator": Tflite({
            "adb_device_id": "5e6fecf",
            "imagenet_accuracy_eval_path": "/data/local/tmp/tf-r2.1-60afa4e/imagenet_accuracy_eval",
            "delegate": "gpu",
            "skip_normalization": True,
            "precision": "F16"
        })
    })
    tester.run()


def model_latency_test():
    from testers.tester_impls.test_model import TestModel
    from testers.inference_sdks.tflite import Tflite
    from testers.sampling.model_sampler import ModelSampler

    tester = TestModel(settings={
        "adb_device_id": "5e6fecf",
        "inference_sdk": Tflite({
            "benchmark_model_path": "/data/local/tmp/tf-r2.1-60afa4e/benchmark_model",
            "su": True
        }),
        "sampler": ModelSampler({
            "model_paths": glob("C:/Users/v-xiat/Microsoft/Shihao Han (FA Talent) - ChannelNas/models/tflite/mnasnet-a1/*.tflite")
        })
    })

    tester.run(benchmark_model_flags={
        "num_runs": 30,
        "use_gpu": False,
    })


if __name__ == '__main__':
    model_latency_test()
