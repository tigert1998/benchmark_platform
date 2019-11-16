import os

import tensorflow as tf

from .inference_sdk import InferenceSdk, InferenceResult
from .utils import concatenate_flags, rfind_assign_float
from testers.utils import adb_push, adb_shell


class Rknn(InferenceSdk):
    @staticmethod
    def generate_model(path, inputs, outputs):
        outputs_ops_names = [o.op.name for o in outputs]

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            constant_graph = tf.graph_util.convert_variables_to_constants(
                sess, sess.graph_def, outputs_ops_names)
            with tf.gfile.FastGFile(os.path.splitext(path)[0] + '.pb', mode='wb') as f:
                f.write(constant_graph.SerializeToString())

            from rknn.api import RKNN

            # take care to modify RKNN.__init__
            rknn = RKNN(verbose=True)
            rknn.config(batch_size=1)
            assert(0 == rknn.load_tensorflow(
                os.path.splitext(path)[0] + '.pb',
                inputs=[i.op.name for i in inputs],
                input_size_list=[i.get_shape().as_list()[1:] for i in inputs],
                # remove batch size
                outputs=outputs_ops_names))
            assert(0 == rknn.build(do_quantization=False,
                                   dataset="", pre_compile=False))
            assert(0 == rknn.export_rknn(path))
            rknn.release()

    @staticmethod
    def fetch_results(adb_device_id, flags) -> InferenceResult:
        model_folder = "/mnt/sdcard/channel_benchmark"
        benchmark_model_folder = "/data/local/tmp/rknn_benchmark_model"
        adb_push(adb_device_id, "model.rknn", model_folder)

        result_str = adb_shell(adb_device_id, "LD_LIBRARY_PATH={}/lib64 {}/rknn_benchmark_model {}".format(
            benchmark_model_folder,
            benchmark_model_folder, concatenate_flags({
                "model_path": "{}/model.rknn".format(model_folder),
                **flags
            })))

        std_ms = rfind_assign_float(result_str, 'std')
        avg_ms = rfind_assign_float(result_str, 'avg')

        # FIXME
        return InferenceResult(avg_ms=avg_ms, std_ms=std_ms, op_profiling=None)
