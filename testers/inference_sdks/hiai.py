import tensorflow as tf
import docker

import os

from .inference_sdk import InferenceSdk, InferenceResult
from .utils import concatenate_flags, rfind_assign_float
from testers.utils import adb_push, adb_shell

container: docker.models.containers.Container =\
    docker.from_env().containers.list()[0]


class Hiai(InferenceSdk):
    def generate_model(self, path, inputs, outputs):
        path = os.path.splitext(path)[0]
        model_basename = os.path.basename(path)

        outputs_ops_names = [o.op.name for o in outputs]

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            constant_graph = tf.graph_util.convert_variables_to_constants(
                sess, sess.graph_def, outputs_ops_names)
            with tf.gfile.FastGFile(path + '.pb', mode='wb') as f:
                f.write(constant_graph.SerializeToString())
            lines = [
                'model_name:{}.cambricon'.format(model_basename),
                'session_run{',
                '  input_nodes({}):'.format(len(inputs)),
                *['  "{}",{}'.format(i.op.name, ','.join(map(str,
                                                             i.get_shape().as_list()))) for i in inputs],
                '  output_nodes({}):'.format(len(outputs)),
                *['  "{}",{}'.format(o.op.name, ','.join(map(str,
                                                             o.get_shape().as_list()))) for o in outputs],
                '}'
            ]
            with open(path + '.txt', 'wb') as f:
                f.write(b'\x0a'.join(map(lambda s: bytes(s, 'ascii'), lines)))

            model_folder = '/'.join(os.getcwd().split(os.path.sep)[-2:])

            convert_cmd = '; '.join([
                'cd /playground/hiai_convert_tools/tools_tensorflow',
                'export LD_LIBRARY_PATH=so',
                './pb_to_offline --graph=../../{}/{}.pb --param_file=../../{}/{}.txt'.format(
                    model_folder, model_basename, model_folder, model_basename),
                'mv {}.cambricon ../../{}'.format(model_basename, model_folder)
            ])
            convert_cmd = "bash -c \"{}\"".format(convert_cmd)

            print("convert_cmd = \"{}\"".format(convert_cmd))
            result = container.exec_run(convert_cmd)
            assert(result.exit_code == 0)
            print(result.output.decode('utf-8'))

    def _fetch_results(self, adb_device_id, model_path, flags) -> InferenceResult:
        model_path = os.path.splitext(model_path)[0]
        model_basename = os.path.basename(model_path)

        model_folder = "/mnt/sdcard/channel_benchmark"
        benchmark_model_folder = "/data/local/tmp/hiai_benchmark_model"
        for ext in ["pb", "txt", "cambricon"]:
            adb_push(adb_device_id,
                     "{}.{}".format(model_path, ext),
                     model_folder)

        result_str = adb_shell(adb_device_id, "LD_LIBRARY_PATH={}/lib64 {}/hiai_benchmark_model {}".format(
            benchmark_model_folder,
            benchmark_model_folder,
            concatenate_flags({
                "offline_model_path": "{}/{}.cambricon".format(model_folder, model_basename),
                "online_model_path": "{}/{}.pb".format(model_folder, model_basename),
                "online_model_parameter": "{}/{}.txt".format(model_folder, model_basename),
                **flags
            })))

        if flags.get("num_runs") is None or flags.get("num_runs") >= 2:
            std_ms = rfind_assign_float(result_str, 'std')
            avg_ms = rfind_assign_float(result_str, 'avg')
        else:
            std_ms = 0
            avg_ms = rfind_assign_float(result_str, 'curr')
        return InferenceResult(avg_ms=avg_ms, std_ms=std_ms, profiling_details=None)
