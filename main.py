from accuracy_testers.accuracy_tester import AccuracyTester

def main():
    tester = AccuracyTester({
        "labels_path": "C:/Users/v-xiat/Downloads/imagenet/val_labels.txt",
        "validation_images_path": "C:/Users/v-xiat/Downloads/imagenet/validation",
        "zip_size": 5000,
        "adb_device_id": "8A9Y0G80H"
    })
    tester.run()


if __name__ == '__main__':
    main()
