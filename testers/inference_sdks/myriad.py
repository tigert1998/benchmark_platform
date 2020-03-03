import os
import re
import shutil

from typing import List

import tensorflow as tf
import numpy as np

from .inference_sdk import InferenceSdk, InferenceResult
from utils.utils import concatenate_flags
from .utils import rfind_assign_float
from utils.connection import Connection

class Myriad(InferenceSdk):
    @staticmethod
    def default_settings():
        return {
            **InferenceSdk.default_settings(),
            # 'openvino_sdk_path': '/home/hanxiao/benchmarks/computer_vision_sdk_2018.5.445',
            'openvino_sdk_path': '/home/hanxiao/benchmark_platform/openvino_2019.2.242',           
            'debug_print': False,
            'api_mode': 'sync',
            'image': '/home/hanxiao/benchmarks/fubuki.jpg',
            'perf_regex': r'([^\s]+)\s+EXECUTED\s+layerType: ([^\s]+)\s+realTime: (\d+)\s+cpu: (\d+)\s+execType: ([\w]+)\n',
            'perf_counter_regex' : r'\[ INFO \] Pefrormance counts for (\d+)-th infer request:\n'
        }
    
    @staticmethod
    def default_flags():
        return {
            **InferenceSdk.default_flags(),
            "niter": 1,
            "nireq": 500,
        }

    def __init__(self, settings={}):
        super().__init__(settings)
        self.openvino_sdk_path = self.settings["openvino_sdk_path"]
        self.debug_print = print if self.settings["debug_print"] == True else lambda x: x
        self.api_mode = self.settings["api_mode"]
        self.image = self.settings["image"]
        self.perf_regex = self.settings["perf_regex"]
        self.perf_counter_regex = self.settings["perf_counter_regex"]

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

            assert 1 == len(inputs)
            
            if os.path.exists(path + '.dir'):
                shutil.rmtree(path + '.dir')

            args = [
                '--input_model %s' % path + '.pb',
                '--output_dir %s'  % path + '.dir',
                '--data_type %s'   % 'FP16',
                '--input_shape %s' % '\({}\)'.format(
                        ','.join(map(str,inputs[0].get_shape().as_list())))
            ]

            args_line = ' '.join(args)

            convert_cmd = '; '.join([
               'python3 %s/deployment_tools/model_optimizer/mo_tf.py %s' %
                            (self.openvino_sdk_path, args_line)
            ])

            convert_cmd = "bash -c \"{}\"".format(convert_cmd)

            self.debug_print("convert_cmd = \"{}\"".format(convert_cmd))

            result = Connection().shell(convert_cmd)
            self.debug_print(result)

            if not os.path.exists(os.path.join(os.path.abspath(path + '.dir'), model_basename + '.xml')):
                if self.debug_print is not print:
                    print('Fatal Error: Failed to generate required file.')
                    print("convert_cmd = \"{}\"".format(convert_cmd))
                    print(result)
                print('Unable to find genereated model,', os.path.join(os.path.abspath(path + '.dir'), model_basename + '.xml'))
                raise FileNotFoundError

    def _fetch_results(self,
                       connection: Connection, model_path,
                       input_size_list: List[List[int]], flags) -> InferenceResult:

        model_path = os.path.splitext(model_path)[0]
        model_basename = os.path.basename(model_path)

        args = [
            '-i %s' % self.image,
            '-m %s' % os.path.join(model_path + '.dir', model_basename + '.xml'),
            '-report_folder %s' % model_path + '.dir',
            '-pc',
            '-d MYRIAD',
            '-report_type detailed_counters',
            '-nireq %d' % flags['nireq'],
            '-niter %d' % flags['niter'],
            '-api %s' % self.api_mode
        ]

        args_line = ' '.join(args)

        convert_cmd = '; '.join([
            'source %s/bin/setupvars.sh' % self.openvino_sdk_path,
            '%s/bin/benchmark_app %s' %
                        (self.openvino_sdk_path, args_line)
        ])

        convert_cmd = "bash -c \"{}\"".format(convert_cmd)

        self.debug_print("convert_cmd = \"{}\"".format(convert_cmd))

        result = Connection().shell(convert_cmd)
        #self.debug_print(result)

        perf_timming = re.findall(self.perf_regex, result)
        perf_counter = re.findall(self.perf_counter_regex, result)

        #self.debug_print('Find %d reports, with %d layers' % (len(perf_counter), len(perf_timming)))

        assert len(perf_timming) % len(perf_counter) == 0

        layer_count = len(perf_timming) // len(perf_counter)

        layerwise_info = []
        tot_stat = []
        layer_stat = {}

        for i_iter in range(len(perf_counter)):
            for i_ireq in range(layer_count):
                layer_time = float(perf_timming[i_iter * layer_count + i_ireq][2]) + float(perf_timming[i_iter * layer_count + i_ireq][3])
                layer_name = '%s_%s' % (perf_timming[i_iter * layer_count + i_ireq][0], perf_timming[i_iter * layer_count + i_ireq][4])

                if perf_timming[i_iter * layer_count + i_ireq][0] != '<Extra>':
                    if len(tot_stat) <= i_iter:
                        tot_stat.append(layer_time)
                    else:
                        tot_stat[i_iter] = tot_stat[i_iter] + layer_time

                if layer_name not in layer_stat:
                    layer_stat[layer_name] = []

                layer_stat[layer_name].append(layer_time)

        for (layer_name, layer_time) in layer_stat.items():  
            layerwise_info.append({
                        "name": layer_name,
                        "time": {
                            'avg_ms': np.mean(layer_time) / 1e3,
                            'std_ms': np.std(layer_time, ddof=1) / 1e3
                            }
                    })
                
        #self.debug_print(tot_stat)
        self.debug_print('Layer Stat, Avg_ms: %.02fms, Std_ms: %.02fms' % (np.mean(tot_stat) / 1e3, np.std(tot_stat, ddof=1) / 1e3))

        return InferenceResult(avg_ms=np.mean(tot_stat) / 1e3, std_ms=np.std(tot_stat, ddof=1) / 1e3, profiling_details=None, layerwise_info=layerwise_info)