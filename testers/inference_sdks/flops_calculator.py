from .inference_sdk import InferenceSdk, InferenceResult
from utils.connection import Connection
from utils.utils import concatenate_flags, rm_ext

from typing import List

import tensorflow as tf
from utils.tf_model_utils import load_graph


class FlopsCalculator(InferenceSdk):
    def generate_model(self, path: str, inputs, outputs):
        outputs_ops_names = [o.op.name for o in outputs]
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            constant_graph = tf.graph_util.convert_variables_to_constants(
                sess, sess.graph_def, outputs_ops_names)
            with tf.gfile.FastGFile(path + '.pb', mode='wb') as f:
                f.write(constant_graph.SerializeToString())

    def _fetch_results(self,
                       connection: Connection, model_path: str,
                       input_size_list: List[List[int]], benchmark_model_flags) -> InferenceResult:
        graph = load_graph(model_path + ".pb")
        with graph.as_default():
            flops = tf.profiler.profile(
                graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
            flops = flops.total_float_ops
        return InferenceResult(avg_ms=None, std_ms=None, profiling_details={"flops": flops}, layerwise_info=None)
