import tensorflow as tf

import os

from .inference_sdk import InferenceSdk, InferenceResult
from .utils import concatenate_flags, rfind_assign_float
from testers.utils import adb_push, adb_shell


class Hiai(InferenceSdk):
    @staticmethod
    def generate_model(path, inputs, outputs):
        outputs_ops_names = [o.op.name for o in outputs]
        model_basename_noext = os.path.splitext(os.path.basename(path))[0]

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            constant_graph = tf.graph_util.convert_variables_to_constants(
                sess, sess.graph_def, outputs_ops_names)
            with tf.gfile.FastGFile(os.path.splitext(path)[0] + '.pb', mode='wb') as f:
                f.write(constant_graph.SerializeToString())
            lines = [
                'model_name:{}'.format(os.path.basename(path)),
                'session_run{',
                '  input_nodes({}):'.format(len(inputs)),
                *['  "{}",{}'.format(i.op.name, ','.join(map(str,
                                                             i.get_shape().as_list()))) for i in inputs],
                '  output_nodes({}):'.format(len(outputs)),
                *['  "{}",{}'.format(o.op.name, ','.join(map(str,
                                                             o.get_shape().as_list()))) for o in outputs],
                '}'
            ]
            with open(os.path.splitext(path)[0] + '.txt', 'wb') as f:
                f.write(b'\x0a'.join(map(lambda s: bytes(s, 'ascii'), lines)))

            model_folder = '/'.join(os.getcwd().split(os.path.sep)[-2:])
            convert_cmd = "docker exec 3fa443ac6087 bash -c \"{}\"".format('; '.join([
                'cd /playground/hiai_convert_tools/tools_tensorflow',
                'export LD_LIBRARY_PATH=so',
                './pb_to_offline --graph=../../{}/{}.pb --param_file=../../{}/{}.txt'.format(
                    model_folder, model_basename_noext, model_folder, model_basename_noext),
                'mv {} ../../{}'.format(os.path.basename(path), model_folder)
            ]))
            print('convert_cmd = "{}"'.format(convert_cmd))
            os.system(convert_cmd)

    @staticmethod
    def fetch_results(adb_device_id, flags) -> InferenceResult:
        model_folder = "/mnt/sdcard/channel_benchmark"
        benchmark_model_folder = "/data/local/tmp/hiai_benchmark_model"
        for ext in ["pb", "txt", "cambricon"]:
            adb_push(adb_device_id, "model." + ext, model_folder)

        result_str = adb_shell(adb_device_id, "LD_LIBRARY_PATH={}/lib64 {}/hiai_benchmark_model {}".format(
            benchmark_model_folder,
            benchmark_model_folder,
            concatenate_flags({
                "offline_model_path": "{}/model.cambricon".format(model_folder),
                "online_model_path": "{}/model.pb".format(model_folder),
                "online_model_parameter": "{}/model.txt".format(model_folder),
                **flags
            })))

        std_ms = rfind_assign_float(result_str, 'std')
        avg_ms = rfind_assign_float(result_str, 'avg')

        return InferenceResult(avg_ms=avg_ms, std_ms=std_ms, op_profiling=None)
