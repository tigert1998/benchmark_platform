from accuracy_tester.accuracy_tester import AccuracyTester
from accuracy_tester.data_preparers.android_data_preparer import AndroidDataPreparer
from accuracy_tester.accuracy_evaluators.tflite import Tflite


def main():
    tester = AccuracyTester({
        "model_paths": [
            "C:/Users/v-xiat/Downloads/imagenet/models/mobilenet_v2_1.0_224_frozen_weight_quant.tflite",
            "C:/Users/v-xiat/Downloads/imagenet/models/mobilenet_v2_1.0_224_frozen_no_quant.tflite",
            "C:/Users/v-xiat/Downloads/imagenet/models/mobilenet_v2_1.0_224_frozen_float16_quant.tflite",
            "C:/Users/v-xiat/Downloads/imagenet/models/mobilenet_v2_1.0_224_frozen_int_quant.tflite",
        ],
        "data_preparer": AndroidDataPreparer({
            "labels_path": "C:/Users/v-xiat/Downloads/imagenet/val_labels.txt",
            "image_id_to_path_func": lambda id: "C:/Users/v-xiat/Downloads/imagenet/validation/ILSVRC2012_val_{0:08}.JPEG".format(id),
            "skip_models_preparation": True,
            "skip_dataset_preparation": True,
            "adb_device_id": "5e6fecf",
        }),
        "accuracy_evaluator": Tflite({
            "adb_device_id": "5e6fecf",
            "imagenet_accuracy_eval_path": "/data/local/tmp/tf-r2.1-60afa4e/imagenet_accuracy_eval",
            "delegate": "gpu"
        })
    })
    tester.run()


if __name__ == '__main__':
    main()
