# TFLite Utilities

This folder contains the code to parse `.tflite` models.
The `schema.fbs` is directly copied from the TensorFlow 2.1 code base.
And the code in `tflite` subfolder is directly generated from `schema.fbs`.
So do NOT change any single line of any of the two above.
Currently, two existing tools in this folder is `model_traverser` and `model_splitter`.
The latter one inherits the former one.

`ModelTraverser` parses a `tflite` model, iterates through all the operators in order of their appearance in the model file and invokes all corresponding hook functions by the type of operators.
The hook functions need to be implemented to accomplish certain goals.
Just as an example, `ModelSplitter` is an implementation of `ModelTraverser`.
It records all operators in the model and then put them into different categories with user-specified criteria (`_op_to_category`).
This criterion can be "always return a constant" which reconstructs the graph
or "return the block name of an operator" which groups operators into blocks. 
Then `construct_tf_graph` can be used to generate a `tf.Graph` by certain category name.

## Sample Usage

```python
model_path = "mobilenet_v2_1.0_224.tflite"
traverser = ModelTraverser(model_path)
traverser.traverse()

import tensorflow as tf
splitter = ModelSplitter(model_path)
def op_to_category(op_detail) -> str:
    return op_detail.outputs[0].split('/')[1]
splitter._op_to_category = op_to_category
splitter.split()
input_tensors, output_tensors = splitter.construct_tf_graph("expanded_conv_2")
```