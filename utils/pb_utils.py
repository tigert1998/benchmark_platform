import tensorflow.compat.v1 as tf
import warnings


def load_graph(frozen_graph_filename):
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")
    return graph


def check_frozen(graph):
    ops = graph.get_operations()
    variable_types = ["VariableV2"]
    variables = list(filter(lambda op: op.type in variable_types, ops))
    if len(variables) >= 1:
        warn_msg = "the input graph has {} possible variable operations".format(
            len(variables))
        warnings.warn(warn_msg)


def analyze_inputs_outputs(graph):
    check_frozen(graph)

    ops = graph.get_operations()
    outputs_set = set(ops)
    inputs = []

    not_input_types = ["Const", "VariableV2"]
    not_output_types = ["Const", "Assign"]

    for op in ops:
        if len(op.inputs) == 0 and (op.type not in not_input_types):
            inputs.append(op)
        else:
            for input_tensor in op.inputs:
                if input_tensor.op in outputs_set:
                    outputs_set.remove(input_tensor.op)
    outputs = list(
        filter(lambda op: op.type not in not_output_types, outputs_set))
    return (inputs, outputs)
