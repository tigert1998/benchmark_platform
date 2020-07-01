# Conversion

This folder contains the code for model conversion
(now it only provides conversion from TensorFlow [saved_model](https://www.tensorflow.org/guide/saved_model) to TFLite).

## to_tflite

[to_tflite.py](to_tflite.py) is a script that accepts a saved_model and then converts it to TFLite format.
It requires the current TensorFlow version to be 2.1.
This script supports all TensorFlow post-training quantization methods,
including [integer quantization](https://www.tensorflow.org/lite/performance/post_training_integer_quant),
[weight quantization](https://www.tensorflow.org/lite/performance/post_training_quant),
[float16 quantization](https://www.tensorflow.org/lite/performance/post_training_float16_quant).

### main

If to_tflite is executed as the main entry, it accepts the following arguments.

|Name|Remark|
|---|---|
|saved_model_path|The path of the saved_model to convert from.|
|quantization|One of "", "weight", "int", "float16". If integer quantization is enabled, the script will use randomly generated images as the calibration dataset.|
|output_path|The path of the TFLite model to convert to.|

### Otherwise

You can import to_tflite as a module. Remember to use TF 2.1 or make sure other versions of TF are also applicable.
Note that you can pass an `imagenet_model_config` object for `int_quant` function to manually configure the calibration
dataset and preprocessing stage of the input model.