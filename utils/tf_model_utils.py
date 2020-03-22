import tensorflow as tf

import warnings
import os
import shutil
import itertools
from functools import reduce
from typing import List, Union, Tuple


def load_graph(frozen_graph_filepath) -> tf.Graph:
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


def calc_graph_mac(graph: tf.Graph):
    ans = 0
    not_involved_op_types = [
        "Identity"
    ]
    involved_op_types = dict()
    for op in graph.get_operations():
        if len(op.inputs) == 0 or len(op.outputs) == 0 or (op.type in not_involved_op_types):
            continue

        if involved_op_types.get(op.type) is None:
            involved_op_types[op.type] = 1
        else:
            involved_op_types[op.type] += 1

        for tensor in itertools.chain(op.inputs, op.outputs):
            shape = tensor.get_shape().as_list()
            if len(shape) >= 1 and not isinstance(shape[0], int):
                shape[0] = 1
            ans += reduce(lambda x, y: x * y, shape, 1)

    print("MAC is collected in these ops: {}".format(involved_op_types))
    return ans


def load_graph_and_fix_shape(
    model_path: str,
    input_op_name: str,
    input_shape: List[int]
) -> tf.Graph:
    assert model_path.endswith(".pb")

    tf.reset_default_graph()
    graph = load_graph(model_path)

    new_graph = tf.Graph()
    with new_graph.as_default():
        input_tensor = tf.placeholder(
            shape=input_shape,
            dtype=tf.float32,
            name=input_op_name
        )
        tf.import_graph_def(
            graph.as_graph_def(), name='',
            input_map={
                input_tensor.name: input_tensor
            }
        )
    return new_graph


def prune_graph(graph: tf.Graph, output_op_names: List[str], output_model_path: str = "model.pb"):
    with tf.Session(graph=graph) as sess:
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,
            tf.get_default_graph().as_graph_def(),
            output_op_names
        )
        with tf.gfile.GFile(output_model_path, "wb") as f:
            f.write(output_graph_def.SerializeToString())
