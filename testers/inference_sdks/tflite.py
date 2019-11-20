import tensorflow as tf

import os

from .inference_sdk import InferenceSdk, InferenceResult
from .utils import concatenate_flags, rfind_assign_float, table_try_float
from testers.utils import adb_push, adb_shell


class Tflite(InferenceSdk):
    def __init__(self, benchmark_model_path, taskset):
        self.benchmark_model_path = benchmark_model_path
        self.taskset = taskset

    @staticmethod
    def generate_model(path, inputs, outputs):
        path = os.path.splitext(path)[0]

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            converter = tf.lite.TFLiteConverter.from_session(
                sess, inputs, outputs)
            tflite_model = converter.convert()
            open(path + '.tflite', 'wb').write(tflite_model)

    def fetch_results(self, adb_device_id: str, model_path: str, flags) -> InferenceResult:
        model_path = os.path.splitext(model_path)[0]
        model_basename = os.path.basename(model_path)

        model_folder = "/mnt/sdcard/channel_benchmark"
        adb_push(adb_device_id, model_path + ".tflite", model_folder)

        taskset_prefix = "" if self.taskset is None else "taskset {}".format(
            self.taskset)

        result_str = adb_shell(adb_device_id, "{} {} {}".format(
            taskset_prefix,
            self.benchmark_model_path,
            concatenate_flags(
                {
                    "graph": "{}/{}.tflite".format(model_folder, model_basename),
                    **flags
                })
        ))

        if flags.get("num_runs") is None or flags.get("num_runs") >= 2:
            std_ms = rfind_assign_float(result_str, 'std') / 1e3
            avg_ms = rfind_assign_float(result_str, 'avg') / 1e3
        else:
            std_ms = 0
            avg_ms = rfind_assign_float(result_str, 'curr') / 1e3

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
