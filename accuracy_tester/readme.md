# Accuracy Tester

An accuracy tester always holds a [AccuracyEvaluator](accuracy_evaluators/readme.md),
a [DataPreparer](data_preparers/readme.md).
It can make accuracy evaluation in the transmission-evaluation-transmission-evaluation-... batching paradigm if no enough hardware storage.
Otherwise, it can also skip the dataset or model preparation and transmission process if the data is already deployed on target devices, and then directly launched the evaluation program.
The data will be dumped into the `test_results` folder.

`AccuracyTester` settings explained:

- `model_details`: A list of ModelDetail objects. The ModelDetail class is explained in the preprocess folder.
- `zip_size`: Accuracy evaluation batch size. If the storage is sufficient, it should be the same with `dataset_size`.
- `dataset_size`: For ImageNet 2012 validation set, it should be 50000.
- `data_preparer`: The data preparer to use.
- `accuracy_evaluator`: The accuracy evaluator to use.