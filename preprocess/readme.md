# Preprocess

## Preprocessor and Preprocess

This folder contains the code for CV model preprocessing.
There exists two classes: `Preprocessor` and `Preprocess`.
The former one processes images through a full pipeline:
reading image from a path,
cropping image and then resizing to a specific size,
normalizing image (subtracting mean and multiplying a scale).
The latter class keeps a `Preprocessor` instance and controls how this `Preprocessor` processes images
(to what stage in the full pipeline the image is processed).

This folder provides two `Preprocessor` with various
available configurations to accommodate manifold model requirements.
The two `Preprocessor`s are `TfPreprocessor` and `TorchPreprocessor` respectively.
The reason behind this division strategy is that we would like the preprocessing to be the exact same with that in the original model repositories,
since TF and PyTorch implementations have some nuances that may probably lead to accuracy bias.
[factory.py](factory.py) predefines the `Preprocess` objects for every model used in the benchmark.
So every time we only need to import the right `Preprocess` and invoke the `execute` method with image path to fulfill a preprocess.

## ModelDetail and Model Archive

`ModelDetail` records information about a model,
including model path, preprocessing function, the input/output node name.
Since a model with its weights usually has multiple formats for different devices
and each format may have several versions (like different quantization methods),
In [model_archive.py](model_archive.py),
`MetaModelDetail` is introduced to keep track a model's "meta-information"
and generate a `ModelDetail` given certain model format and version.
Hence, `MetaModelDetail` also records available formats and hardware for filtering demands
besides the basic information in `ModelDetail`.

`meta_model_details` is a predefined model list for further usage.
Also, a `get_model_details` is introduced to easily filter `meta_model_details` by model's name, format, version and hardware to use.
It is an extremely simple and naive implementation and you may change it if a better design is emerging.
`get_model_details` returns a list of `ModelDetail` and accepts the following parameters:

- `model_names`: A list of strings. A model will be returned only if its name has an overlap with one of  `model_names`. If `None` is supplied, all models will be legit.
- `model_format`: The model format to chose.
- `versions`: A list of strings. The chosen versions.
- `hardware`: The hardware that the model will run on (also used for filtering).
