import tensorflow as tf

import warnings
import os
import shutil
from typing import List, Union, Tuple


def load_graph(frozen_graph_filepath):
    with tf.compat.v1.gfile.GFile(frozen_graph_filepath, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")
    return graph


def to_saved_model(
    sess,
    inputs: List[Union[tf.Tensor, tf.Operation]],
    outputs: List[Union[tf.Tensor, tf.Operation]],
    path: str,
    replace_original_dir: bool
):
    from tensorflow.python.saved_model import signature_constants
    from tensorflow.python.saved_model import tag_constants

    if replace_original_dir:
        if os.path.isdir(path):
            shutil.rmtree(path)

    inputs_dic = {
        "input_{}".format(idx): i if isinstance(i, tf.Tensor) else i.outputs[0]
        for idx, i in zip(range(len(inputs)), inputs)
    }
    outputs_dic = {
        "output_{}".format(idx): o if isinstance(o, tf.Tensor) else o.outputs[0]
        for idx, o in zip(range(len(outputs)), outputs)
    }

    builder = tf.saved_model.builder.SavedModelBuilder(path)
    sigs = {}
    sigs[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = \
        tf.saved_model.signature_def_utils.predict_signature_def(
        inputs_dic, outputs_dic
    )
    builder.add_meta_graph_and_variables(
        sess,
        [tag_constants.SERVING],
        signature_def_map=sigs
    )

    builder.save()


def check_frozen(graph):
    ops = graph.get_operations()
    variable_types = ["VariableV2"]
    variables = list(filter(lambda op: op.type in variable_types, ops))
    if len(variables) >= 1:
        warn_msg = "the input graph has {} possible variable operations".format(
            len(variables))
        warnings.warn(warn_msg)


def analyze_inputs_outputs(graph) -> Tuple[List[tf.Operation], List[tf.Operation]]:
    check_frozen(graph)

    ops = graph.get_operations()
    outputs_set = set(ops)
    inputs = []

    input_types = [
        "Placeholder"
    ]
    not_output_types = [
        "Const",
        "Assign",
    ]

    for op in ops:
        if len(op.inputs) == 0 and (op.type in input_types) and (len(op.outputs) >= 1):
            inputs.append(op)
        else:
            for input_tensor in op.inputs:
                if input_tensor.op in outputs_set:
                    outputs_set.remove(input_tensor.op)
    outputs = list(
        filter(lambda op: (op.type not in not_output_types) and (len(op.outputs) >= 1), outputs_set))
    return (inputs, outputs)
