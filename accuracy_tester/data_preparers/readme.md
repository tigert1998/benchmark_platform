# Data Preparer

In many cases, accuracy testing needs an explicit dataset transmission process.
For example, a validation dataset and models must be transmitted to a SoC before launching a benchmark.
If storage on the target hardware is not sufficient,
a transmission-evaluation-transmission-evaluation-... batching paradigm must be adopted to avoid overflow.
Therefore, a `DataPreparer` is required to control the evaluation pace.

`DataPreparer` methods and explanations:

|Method|Access Control (Pseudo)|Remark|
|-|-|-|
|image_path_label_gen|Public|A generator. Generate a series of images with labels in the order specified in `labels_path`.|
|_prepare_models|Protected|Overrode to customize model preparation.|
|prepare_models|Public|Prepare a list of models and transmit to a remote destination.|
|_prepare_dateset|Protected|Overrode to customize dataset preparation.|
|prepare_dateset|Public|Pack a range of images with their labels and transmit to a remote destination.|

`DataPreparer` settings explanations:

- `labels_path`: The path of the label file. This label file contains `n` lines where `n` is dataset size.
Each line has two parts: the image path and a corresponding label which is the index of output logit ID.
- `validation_set_path`: Validation set folder path. The folder contains the images.
- `skip_models_preparation`: If enabled as "true", `prepare_models` will definitely be skipped.
- `skip_dataset_preparation`: If enabled as "true", `prepare_dateset` will definitely be skipped. 

