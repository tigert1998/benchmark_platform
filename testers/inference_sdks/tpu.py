from .inference_sdk import InferenceSdk, InferenceResult
from utils.connection import Connection
from utils.utils import concatenate_flags, rm_ext
from .utils import rfind_assign_float

import numpy as np
import os
import tensorflow as tf


class Tpu(InferenceSdk):
    def generate_model(self, path, inputs, outputs):
        path = rm_ext(path)
        assert len(inputs) == 1

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            converter = tf.lite.TFLiteConverter.from_session(
                sess, inputs, outputs)

        def representative_data_gen():
            yield [np.random.randint(0, 256, inputs[0].get_shape().as_list()).astype(np.float32)]

        converter.representative_dataset = representative_data_gen
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
        tflite_model = converter.convert()
        open(path + '.tflite', 'wb').write(tflite_model)

    def _fetch_results(self, connection: Connection, model_path: str, input_size_list, benchmark_model_flags) -> InferenceResult:
        model_basename = os.path.basename(model_path)
        model_folder = "/data"
        connection.push(model_path + ".tflite", model_folder)

        cmd = "{} run /home/tigertang/edgetpu_profiling/main.py {}".format(
            "/opt/anaconda3/bin/conda",
            concatenate_flags({
                "model_path": "{}/{}.tflite".format(model_folder, model_basename),
                **benchmark_model_flags
            })
        )
        print(cmd)
        result_str = connection.shell(cmd)
        avg_ms = rfind_assign_float(result_str, "avg")
        std_ms = rfind_assign_float(result_str, "std")
        return InferenceResult(avg_ms=avg_ms, std_ms=std_ms, profiling_details=None, layerwise_info=None)
