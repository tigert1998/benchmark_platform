# Utils

This folder contains various common utilities used by other modules.

## ClassWithSettings

The class with `self.settings` as a mandatory parameter.
All its subclasses can use the `snapshot` and `brief` function to generate an "abstract" of their instances.
This class is designed for readable instance serialization to json or a simple string. 
Note that `self.settings` must be a subset of `self.default_settings()` which should be overrode by the subclasses,
or an assertion error will be thrown in the init stage.

|Method|Remark|
|---|---|
|\_\_init\_\_|`settings` should be a dict. The `settings` should be a subset of `self.default_settings()`.|
|snapshot|Returns `self.settings`. `ClassWithSettings` subclasses in `self.settings` will be recursively replaced with their own `self.settings`.|
|brief|Generates a string as "abstract". `ClassWithSettings` subclasses in `self.settings` will be recursively embedded into the brief string.|
|default_settings|Returns a dict, representing the default settings.|

## Connection

The connection class is the common interface for various connection types.
Two available implementations are `Ssh` and `Adb`.

## CSVWriter

A normal CSV only permits the first row as the title. It makes logging data points with different schemes harder.
This `CSVWriter` allows logging data points with different titles (schemes).
`update_data` is the only public method except `__init__`.
It accepts 3 parameters:

- `filename`: Path of the file to log. A new title will be written if a brand new file is present.
- `data`: A dict that maps keys (titles) to their values. If the keys (titles) are different from the last data point, then the titles will be logged as a row.
- `is_resume`: If set as `true`, the file will be opened in append mode.

## TF Model Utils

|Function|Remark|
|-|-|
|freeze_graph|Accepts a `tf.Graph` and the output operator names. Returns a new `tf.Graph` as the frozen graph (variables frozen as constants).|
|load_graph|Loads a frozen pb from disk and converts to `tf.Graph`.|
|load_graph_with_normalization|Loads a frozen pb from disk and appends additional normalization (subtract `mean` and divide `std`) operators in the front of the pb.|
|pad_graph|Accepts a `tf.Graph` and then generates a new graph with new operators that are specified with `pad_before_input` and `pad_after_output`.|
|to_saved_model|Convert current TF session graph to saved_model format.|
|analyze_inputs_outputs|Analyzes a `tf.Graph` and then returns possible inputs and outputs.|
|calc_graph_mac|Calculate the MAC (memory access cost) of a `tf.Graph`.|
|load_graph_and_fix_shape|Loads a frozen pb from disk and fix the input shape of the graph.|
|prune_graph|Prune a `tf.Graph` (remove unused operators, constant folding, etc).|