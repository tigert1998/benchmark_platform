import tensorflow as tf
import numpy as np

import warnings
import os
import shutil
import itertools
from functools import reduce
from typing import List, Union, Tuple, Callable
import logging


def load_graph(frozen_graph_filepath: str) -> tf.Graph:
    with tf.compat.v1.gfile.GFile(frozen_graph_filepath, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")
    return graph


def load_graph_with_normalization(
    frozen_graph_filepath: str,
    input_op_name: str, output_op_name: str,
    mean: List[float], std: List[float]
) -> tf.Graph:
    assert len(mean) == 3 and len(std) == 3

    graph = load_graph(frozen_graph_filepath)
    origin_input_shape = graph.get_operation_by_name(
        input_op_name
    ).outputs[0].get_shape().as_list()
    imsize = origin_input_shape[1]
    assert imsize == origin_input_shape[2] and origin_input_shape[3] == 3

    with tf.Graph().as_default() as new_graph:
        input_tensor = tf.placeholder(
            shape=[1, imsize, imsize, 3],
            dtype=tf.float32,
            name=input_op_name
        )
        normalized = tf.math.add(
            input_tensor, -np.array(mean).astype(np.float32))
        normalized = tf.math.multiply(
            normalized, 1 / np.array(std).astype(np.float32))

        tf.import_graph_def(graph.as_graph_def(), name="", input_map={
            input_op_name: normalized
        })

    return prune_graph(new_graph, [input_op_name], [output_op_name])


def pad_graph(
    graph: tf.Graph,
    input_tensor_names: List[str],
    output_tensor_names: List[str],
    pad_before_input: Callable[[List[List[int]]], Tuple[List[tf.Tensor], List[tf.Tensor]]],
    pad_after_output: Callable[[List[tf.Tensor]], List[tf.Tensor]]
) -> Tuple[tf.Graph, List[str], List[str]]:
    graph_def = graph.as_graph_def()
    input_tensor_shapes = [
        graph.get_tensor_by_name(name).get_shape().as_list()
        for name in input_tensor_names
    ]

    with tf.Graph().as_default() as new_graph:
        input_tensors, nets = pad_before_input(input_tensor_shapes)

        tf.import_graph_def(graph_def, name="padded", input_map={
            name: nets[i]
            for i, name in enumerate(input_tensor_names)
        })

        nets = [
            new_graph.get_tensor_by_name(f"padded/{name}")
            for name in output_tensor_names
        ]
        output_tensors = pad_after_output(nets)
        input_tensor_names = [i.name for i in input_tensors]
        output_tensor_names = [i.name for i in output_tensors]

        return new_graph, input_tensor_names, output_tensor_names


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

    if tf.__version__.startswith("1."):
        saved_model = tf.saved_model
    else:
        saved_model = tf.compat.v1.saved_model

    builder = saved_model.builder.SavedModelBuilder(path)
    sigs = {}
    sigs[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = \
        saved_model.signature_def_utils.predict_signature_def(
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
        # I think unnecessary
        "Identity",
        "Shape",
        "Squeeze",
        "Gather",
        "GatherV2",

        # activations
        "BiasAdd",
        "Relu",
        "Relu6",
        "Sigmoid",
        "FusedBatchNorm",
        "FusedBatchNormV2",
        "FusedBatchNormV3",
    ]
    involved_op_types = {
        # compute
        "Conv2D",
        "DepthwiseConv2dNative",
        "AvgPool",
        "MatMul",
        "MaxPool",
        "Softmax",

        "Split",
        "Concat",
        "ConcatV2",
        "Reshape",
        "Transpose",
        "Pad",
        "Slice",
        "StridedSlice",
    }
    oneside_elewise_op_types = {
        "Add",
        "AddV2",
        "Mul"
    }
    for op in graph.get_operations():
        if len(op.inputs) == 0 or len(op.outputs) == 0:
            continue
        if op.type in not_involved_op_types:
            continue
        if op.type in oneside_elewise_op_types:
            assert len(op.inputs) == 2
            cnt = sum(map(
                lambda tensor: int(tensor.op.type == "Const"),
                op.inputs
            ))
            print("Found 1x {} that takes {} constants as inputs".format(op.type, cnt))
            if cnt >= 1:
                continue

        if op.type not in involved_op_types.union(oneside_elewise_op_types):
            logging.fatal("{} not in involved op types".format(op.type))
            exit(0)

        for tensor in itertools.chain(op.inputs, op.outputs):
            if tensor.get_shape()._dims is None:
                warnings.warn("Not calculated: {}".format(tensor))
                continue
            else:
                shape = tensor.get_shape().as_list()
            if len(shape) >= 1 and not isinstance(shape[0], int):
                shape[0] = 1
            ans += reduce(lambda x, y: x * y, shape, 1)

    return ans


def load_graph_and_fix_shape(
    model_path: str,
    input_op_name: str,
    input_shape: List[int],
    output_op_name: str
) -> tf.Graph:
    from tensorflow.tools.graph_transforms import TransformGraph

    assert model_path.endswith(".pb")

    graph = load_graph(model_path)

    new_graph_def = TransformGraph(
        graph.as_graph_def(),
        [input_op_name], [output_op_name],
        [
            'strip_unused_nodes(type=float, shape="{}")'.format(
                ','.join(map(str, input_shape))
            )
        ]
    )

    with tf.Graph().as_default() as new_graph:
        input_tensor = tf.placeholder(
            shape=input_shape,
            dtype=tf.float32,
            name=input_op_name
        )
        tf.import_graph_def(new_graph_def, name="", input_map={
            input_tensor.name: input_tensor
        })

    return new_graph


def prune_graph(
    graph: tf.Graph,
    input_op_names: List[str],
    output_op_names: List[str],
) -> tf.Graph:
    from tensorflow.tools.graph_transforms import TransformGraph

    transforms = [
        'remove_nodes(op=Identity)',
        'strip_unused_nodes',
        'fold_constants(ignore_errors=false)',
        'fold_batch_norms'
    ]

    optimized_graph_def = TransformGraph(
        graph.as_graph_def(), input_op_names, output_op_names, transforms
    )

    with tf.Graph().as_default() as new_graph:
        tf.import_graph_def(optimized_graph_def, name="")
        return new_graph
