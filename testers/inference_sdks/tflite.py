import tensorflow as tf

import os

from .inference_sdk import InferenceSdk, InferenceResult
from .utils import concatenate_flags, rfind_assign_float, table_try_float, rfind_assign_int
from testers.utils import adb_push, adb_shell


class Tflite(InferenceSdk):
    @staticmethod
    def default_settings():
        return {
            "benchmark_model_path": None,
            "taskset": None,
            "su": False
        }

    @staticmethod
    def default_flags():
        return {
            "use_gpu": None
        }

    def generate_model(self, path, inputs, outputs):
        path = os.path.splitext(path)[0]

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            converter = tf.lite.TFLiteConverter.from_session(
                sess, inputs, outputs)
            tflite_model = converter.convert()
            open(path + '.tflite', 'wb').write(tflite_model)

    def _launch_benchmark(self, adb_device_id: str, model_path: str, flags):
        model_path = os.path.splitext(model_path)[0]
        model_basename = os.path.basename(model_path)

        model_folder = "/mnt/sdcard/channel_benchmark"
        adb_push(adb_device_id, model_path + ".tflite", model_folder)

        if self.settings["taskset"] is None:
            taskset_prefix = ""
        else:
            taskset_prefix = "taskset " + self.settings["taskset"].strip()

        cmd = "{} {} {}".format(
            taskset_prefix,
            self.settings["benchmark_model_path"],
            concatenate_flags(
                {
                    "graph": "{}/{}.tflite".format(model_folder, model_basename),
                    **flags
                })
        )
        print(cmd.strip())
        return adb_shell(adb_device_id, cmd, self.settings["su"])

    def _fetch_results(self, adb_device_id: str, model_path: str, flags) -> InferenceResult:
        result_str = self._launch_benchmark(adb_device_id, model_path, flags)

        if rfind_assign_int(result_str, 'count') >= 2:
            std_ms = rfind_assign_float(result_str, 'std') / 1e3
            avg_ms = rfind_assign_float(result_str, 'avg') / 1e3
        else:
            std_ms = 0
            avg_ms = rfind_assign_float(result_str, 'curr') / 1e3

        use_delegate = flags.get("use_nnapi", False) or flags.get(
            "use_gpu", False) or flags.get("use_legacy_nnapi", False)
        enable_op_profiling = flags.get("enable_op_profiling", False)

        if use_delegate or (not enable_op_profiling):
            return InferenceResult(avg_ms=avg_ms, std_ms=std_ms, profiling_details=None)
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
            return InferenceResult(avg_ms=avg_ms, std_ms=std_ms, profiling_details=op_profiling)
