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
        "model_paths": glob("C:/Users/v-xiat/Downloads/imagenet/models/*.tflite"),
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
            "delegate": "cpu",
            "skip_normalization": True
        })
    })
    tester.run()


def latency_test_441_to_444():
    import testers.tester_impls.test_conv_gpu_mem_comp_split
    import testers.inference_sdks.tflite_modified
    import testers.sampling.conv_sampler

    tester = testers.tester_impls.test_conv_gpu_mem_comp_split.TestConvGpuMemCompSplit(
        adb_device_id="5e6fecf",
        inference_sdk=testers.inference_sdks.tflite_modified.TfliteModified({
            "benchmark_model_path": "/data/local/tmp/tf-r2.1-60afa4e/benchmark_model_modified",
            "su": True
        }),
        sampler=testers.sampling.conv_sampler.ConvSampler({
            "filter": lambda sample: sample[6] == 960 and sample[-1] == 1
        }))

    tester.run(settings={}, benchmark_model_flags={
        "num_runs": 30,
        "use_gpu": True,
        "gpu_precision_loss_allowed": False,
        "work_group_size": "4,4,4"
    })


if __name__ == '__main__':
    accuracy_test_tflite()
