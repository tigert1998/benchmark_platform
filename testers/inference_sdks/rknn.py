import os

import tensorflow as tf
import numpy as np

from rknn.api import RKNN

from .inference_sdk import InferenceSdk, InferenceResult
from .utils import rfind_assign_float
from utils.utils import adb_push, adb_shell, concatenate_flags
from utils.stat import Stat


class Rknn(InferenceSdk):
    @staticmethod
    def default_settings():
        return {
            **InferenceSdk.default_settings(),
            "rknn_target": "rk1808",
        }

    @staticmethod
    def default_flags():
        return {
            **InferenceSdk.default_flags(),
            "num_runs": 50,
        }

    def generate_model(self, path, inputs, outputs):
        path = os.path.splitext(path)[0]

        outputs_ops_names = [o.op.name for o in outputs]

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            constant_graph = tf.graph_util.convert_variables_to_constants(
                sess, sess.graph_def, outputs_ops_names)
            with tf.gfile.FastGFile(path + '.pb', mode='wb') as f:
                f.write(constant_graph.SerializeToString())

            # remember to modify RKNN.__init__
            rknn = RKNN(verbose=True)
            rknn.config(batch_size=1)
            assert 0 == rknn.load_tensorflow(
                path + '.pb',
                inputs=[i.op.name for i in inputs],
                input_size_list=[i.get_shape().as_list()[1:] for i in inputs],
                # remove batch size
                outputs=outputs_ops_names)
            assert 0 == rknn.build(do_quantization=False,
                                   dataset="", pre_compile=False)
            assert 0 == rknn.export_rknn(path + '.rknn')
            rknn.release()

    def _fetch_results_on_soc(self, adb_device_id, model_path, input_size_list, flags) -> InferenceResult:
        model_path = os.path.splitext(model_path)[0]
        model_basename = os.path.basename(model_path)

        model_folder = "/mnt/sdcard/channel_benchmark"
        benchmark_model_folder = "/data/local/tmp/rknn_benchmark_model"
        adb_push(adb_device_id, model_path + ".rknn", model_folder)

        result_str = adb_shell(adb_device_id, "LD_LIBRARY_PATH={}/lib64 {}/rknn_benchmark_model {}".format(
            benchmark_model_folder,
            benchmark_model_folder, concatenate_flags({
                "model_path": "{}/{}.rknn".format(model_folder, model_basename),
                **flags
            })))

        if flags.get["num_runs"] >= 2:
            std_ms = rfind_assign_float(result_str, 'std')
            avg_ms = rfind_assign_float(result_str, 'avg')
        else:
            std_ms = 0
            avg_ms = rfind_assign_float(result_str, 'curr')

        # FIXME
        return InferenceResult(avg_ms=avg_ms, std_ms=std_ms, profiling_details=None, layerwise_info=None)

    def _fetch_results_with_py_api(self, adb_device_id, model_path, input_size_list, flags) -> InferenceResult:
        rknn = RKNN()

        assert 0 == rknn.load_rknn(model_path + ".rknn")
        assert 0 == rknn.init_runtime(
            target=self.settings["rknn_target"], perf_debug=True)

        if input_size_list is None:
            input_size_list = [1, 299, 299, 3]
        image = np.random.rand(*input_size_list).astype(np.float32)

        stat = Stat()
        layerwise_info = None
        for i in range(flags["num_runs"]):
            result = rknn.eval_perf(inputs=[image], is_print=False)
            stat.update(result["total_time"] / 1e3)
            result_layer = result["layers"]
            if layerwise_info is None:
                layerwise_info = result_layer
                for value in layerwise_info.values():
                    t = value["time"]
                    value["time"] = Stat()
                    value["time"].update(t / 1e3)
            else:
                for key, value in result_layer.items():
                    layerwise_info[key]["time"].update(value["time"] / 1e3)

        for value in layerwise_info.values():
            layer_stat = value.pop("time")
            value["time"] = {
                "avg_ms": layer_stat.avg(),
                "std_ms": layer_stat.std()
            }

        rknn.release()

        return InferenceResult(avg_ms=stat.avg(), std_ms=stat.std(), profiling_details=None, layerwise_info=layerwise_info)

    def _fetch_results(self, adb_device_id, model_path, input_size_list, flags) -> InferenceResult:
        if self.settings["rknn_target"] is None:
            return self._fetch_results_on_soc(adb_device_id, model_path, input_size_list, flags)
        else:
            return self._fetch_results_with_py_api(adb_device_id, model_path, input_size_list, flags)
