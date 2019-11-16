import tensorflow as tf

from .inference_sdk import InferenceSdk, InferenceResult
from .utils import concatenate_flags, rfind_assign_float, table_try_float
from ..utils import adb_push, adb_shell


class Tflite(InferenceSdk):
    @staticmethod
    def generate_model(path, inputs, outputs):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            converter = tf.lite.TFLiteConverter.from_session(
                sess, inputs, outputs)
            tflite_model = converter.convert()
            open(path, 'wb').write(tflite_model)

    @staticmethod
    def fetch_results(adb_device_id, flags) -> InferenceResult:
        """push model to an android device and fetch results
        Args:
            adb_device_id: adb device ID
            flags: Flag dict for benchmark_model

        Returns:
            InferenceResult
        """

        model_folder = "/mnt/sdcard/channel_benchmark"
        adb_push(adb_device_id, "model.tflite", model_folder)

        result_str = adb_shell(adb_device_id, "taskset f0 /data/local/tmp/master-20191015/benchmark_model {}".format(
            concatenate_flags(
                {
                    "graph": "{}/model.tflite".format(model_folder),
                    **flags
                })
        ))

        std_ms = rfind_assign_float(result_str, 'std') / 1e3
        avg_ms = rfind_assign_float(result_str, 'avg') / 1e3

        use_delegate = flags.get("use_nnapi", False) or flags.get(
            "use_gpu", False) or flags.get("use_legacy_nnapi", False)
        enable_op_profiling = flags.get("enable_op_profiling", False)

        if use_delegate or (not enable_op_profiling):
            return InferenceResult(avg_ms=avg_ms, std_ms=std_ms, op_profiling=None)
        else:
            op_profiling = []
            started = False
            for line in result_str.split('\n'):
                if "Top by Computation Time" in line:
                    break
                if "Run Order" in line:
                    started = True
                    continue
                if started:
                    cells = list(filter(lambda x: len(x) >= 1, map(
                        lambda x: x.strip(), line.split('\t'))))
                    if len(cells) == 0:
                        continue
                    op_profiling.append(cells)
            op_profiling = table_try_float(op_profiling)
            return InferenceResult(avg_ms=avg_ms, std_ms=std_ms, op_profiling=op_profiling)
