from accuracy_tester.accuracy_tester import AccuracyTester


def main():
    tester = AccuracyTester({
        "labels_path": "C:/Users/v-xiat/Downloads/imagenet/val_labels.txt",
        "validation_images_path": "C:/Users/v-xiat/Downloads/imagenet/validation",
        "model_paths": [
            "C:/Users/v-xiat/Downloads/imagenet/models/mobilenet_v2_1.0_224_frozen_weight_quant.tflite",
            "C:/Users/v-xiat/Downloads/imagenet/models/mobilenet_v2_1.0_224_frozen_no_quant.tflite",
            "C:/Users/v-xiat/Downloads/imagenet/models/mobilenet_v2_1.0_224_frozen_float16_quant.tflite",
            "C:/Users/v-xiat/Downloads/imagenet/models/mobilenet_v2_1.0_224_frozen_int_quant.tflite",
        ],
        "skip_data_preparation": True,
        "zip_size": 50000,
        "adb_device_id": "5e6fecf"
    })
    tester.run()


if __name__ == '__main__':
    main()
