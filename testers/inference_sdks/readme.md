# Inference SDK

Inference SDK is a python wrapper for common inference frameworks.
In this folder, `InferenceSdk` in [inference_sdk.py](inference_sdk.py) is the root class.
It abstracts two actions: generate a model and benchmark on the edge device.
These two actions correspond to `generate_model` and `_fetch_results` respectively.

## Generating a Model

`generate_model` takes the following arguments:
- `path`: The path of the model to generate.
Path extension will be stripped out to keep consistent across multiple different inference SDKs.
For instance, if the model path is `path/to/model.pb`, then only `path/to/model` will be passed as an argument.
- `inputs`: The input tensors in the current default TF graph.
- `outputs`: The output tensors in the current default TF graph.

## Fetch Benchmark Results

`_fetch_results` pushes a model to a general device, runs the benchmark and at last fetches the results.
It takes the following parameters:

- `connection`: A `Connection` object that is the abstraction of the device.
The `Connection` object is specified in the "utils" folder.
- `model_path`: Model path without extension.
- `input_size_list`: Inputs shapes of the model.
- `flags`: The flags dict for this benchmark framework.