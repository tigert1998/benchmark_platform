from .inference_sdk import InferenceSdk, InferenceResult
from utils.connection import Connection
from utils.utils import concatenate_flags, rm_ext

from typing import List

import tensorflow as tf
from utils.tf_model_utils import load_graph, calc_graph_mac, analyze_inputs_outputs


class FlopsCalculator(InferenceSdk):
    def generate_model(self, path: str, inputs, outputs):
        outputs_ops_names = [o.op.name for o in outputs]
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            constant_graph = tf.graph_util.convert_variables_to_constants(
                sess, sess.graph_def, outputs_ops_names)
            with tf.gfile.FastGFile(path + '.pb', mode='wb') as f:
                f.write(constant_graph.SerializeToString())

    def _fetch_results(
        self,
        connection: Connection, model_path: str,
        input_size_list: List[List[int]],
        benchmark_model_flags
    ) -> InferenceResult:
        graph = load_graph(model_path + ".pb")
        with graph.as_default():
            flops = tf.profiler.profile(
                graph,
                options=tf.profiler.ProfileOptionBuilder.float_operation(),
            )
            flops = flops.total_float_ops
            mac = calc_graph_mac(graph)

        # deduce output shape
        _, outputs = analyze_inputs_outputs(graph)
        assert len(outputs) == 1
        output_shape = outputs[0].outputs[0].get_shape().as_list()
        if len(output_shape) == 4:
            output_imsize = output_shape[1]
            assert output_imsize == output_shape[2]
            cout = output_shape[-1]
        elif len(output_shape) == 2:
            output_imsize = 1
            cout = output_shape[-1]
        else:
            assert False

        return InferenceResult(
            avg_ms=None, std_ms=None,
            profiling_details={
                "flops": flops,
                "mac": mac,
                "output_imsize": output_imsize,
                "cout": cout
            },
            layerwise_info=None
        )
