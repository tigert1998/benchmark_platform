# Accuracy Evaluator

This folder contains the code for various ML model accuracy evaluators.
Now it only supports evaluating ImageNet 2012 CV models.
`AccuracyEvaluatorDef` is the root interface class and all other evaluators must be its subclass.
Its only public method is `evaluate_models` which needs two parameters:
`model_details` and `image_path_label_gen`.
The former one is the list of models to benchmark.
Note that `ModelDetail` type is specified in the preprocess folder.
The latter parameter is a python generator that outputs an image path and an image label as one tuple at a time.
The image label must be a stringified integer ranging from 0 to 999/1000, according to the number of model outputs.
Also note that if the model gives 1001 outputs, the 0th logit must be the background.

The followings are overview for evaluator implementations:

- [onnx.py](onnx.py): Evaluates with ONNX Runtime framework.
- [rknn.py](rknn.py)
- [tf_evaluator.py](tf_evaluator.py): Evaluates TF frozen pb graphs.
- [tflite.py](tflite.py): Evaluates TFLite models.
- [tpu.py](tpu.py): Evaluates models on EdgeTPU.
