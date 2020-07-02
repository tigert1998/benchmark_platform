# Tester

A `Tester` instance makes one test at a time and logs the data point to a CSV file through `CSVWriter`.
The data points are sampled from `Sampler` which is implemented in [sampling](sampling/readme.md).
Also, an `InferenceSdk` ([inference_sdks](inference_sdks/readme.md)) is set in the settings of `Tester` to configure the framework to use.
Besides that, a `Connection` object is used to record device information, specifying the data transmission process.
A benchmarking task usually takes `Tester` an extremely long time to finish.
Therefore, it is crucial to enable checkpoint resuming once the process is interrupted.
The `resume_from` configuration item is specially tailored for this scenario.
Once set as **the last executed data point**, the `Tester` will continue from this data point and benchmark next ones.

A `Tester` implementation should override the `_test_sample` method,
which returns the benchmarked `InferenceResult` for each sample.
The `InferenceResult` contains the following attributes:

- `avg_ms`: Average inference time. Usually used to represent the whole model inference time.
- `std_ms`: The standard variance of the model inference time.
- `layerwise_info`: The layerwise inference latency. A list of dict which has the schema:
```python
{
    "name": "sample_layer_name",
    "time": {
        "avg_ms": 1, # average layer inference latency
        "std_ms": 0 # standard variance of layer inference latency
    }
}
```
- `profiling_details`: Other information that you would like to log as well.

The `tester_impls` folder contains multiple predefined benchmarking tasks.
There are two root classes.
One is `TestSingleLayer` and another is `TestStacked`.
The former one is the root class for single-layer model testers.
It features a `_pad_before_input` and a `_pad_after_output` function
which can be used to generate additional operator in the front of/in the tail of the single-layer model
for removing some unexpected inference overhead (now only used on RKNN).
The latter one can stack several exact same layers together to extract the single-layer latency.