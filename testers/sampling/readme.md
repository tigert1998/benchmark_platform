# Sampler

This folder contains the code for all data point samplers.
`Sampler` is the root class and other classes should be its subclasses.
Each `Sampler` generates a series of data points with a fixed schema to benchmark.
The methods to override in `Sampler` is
`get_sample_titles`, `_get_samples_without_filter` and `_get_serializable_sample`.
The first one returns the schema title list.
The second one should be a Python generator that generates the samples to benchmark.
The last one only needs to be overridden if the sample list contains types that cannot be converted to string.