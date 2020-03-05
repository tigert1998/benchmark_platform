import os
import re
import shutil

import math

from typing import List

import tensorflow as tf
import numpy as np

from .inference_sdk import InferenceSdk, InferenceResult
from utils.utils import concatenate_flags
from .utils import rfind_assign_float
from utils.connection import Connection

class Snpe(InferenceSdk):
    @staticmethod
    def default_settings():
        return {
            **InferenceSdk.default_settings(),
            'snpe_sdk_path': '/home/hanxiao/benchmarks/snpe-1.30.0.480',         
            'tensorflow_path': '/home/hanxiao/.local/lib/python3.6/site-packages/tensorflow',
            'bench_regex': r',,(layer_\d+) \(Name:([^\s]+) Type:([\w]+)\),(\d+),(\d+),(\d+),([^\n]+)',
            'debug_print': False,
        }
    
    @staticmethod
    def default_flags():
        return {
            **InferenceSdk.default_flags(),
            'runtimes': ['DSP'],
            'performance_mode': 'sustained_high_performance'
        }

    def __init__(self, settings={}):
        super().__init__(settings)
        self.snpe_sdk_path = self.settings["snpe_sdk_path"]
        self.tensorflow_path = self.settings["tensorflow_path"]
        self.bench_regex = self.settings["bench_regex"]
        self.debug_print = print if self.settings["debug_print"] == True else lambda x: x


    def generate_model(self, path, inputs, outputs):
        model_basename = os.path.basename(path)

        outputs_ops_names = [o.op.name for o in outputs]

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            constant_graph = tf.graph_util.convert_variables_to_constants(
                sess, sess.graph_def, outputs_ops_names)
            with tf.gfile.FastGFile(path + '.pb', mode='wb') as f:
                f.write(constant_graph.SerializeToString())

            assert 1 == len(inputs)
            assert 1 == len(outputs_ops_names)
            
            if os.path.exists(path + '.dir'):
                shutil.rmtree(path + '.dir')

            args = [
                '--input_network %s' % os.path.abspath(path + '.pb'),
                '--output_path %s'  % os.path.abspath(path + '.pb').replace('.pb', '.dlc'),
                '--input_dim "%s" "%s"' % (inputs[0].op.name,
                        ','.join(map(str,inputs[0].get_shape().as_list()))),
                '--out_node %s' % outputs_ops_names[0],
                '--allow_unconsumed_nodes'
            ]

            args_line = ' '.join(args)

            convert_cmd = '; '.join([
                'cd %s' % self.snpe_sdk_path,
                'source %s/bin/envsetup.sh -t %s' % (self.snpe_sdk_path, self.tensorflow_path),
                'python %s/bin/x86_64-linux-clang/snpe-tensorflow-to-dlc %s' % (self.snpe_sdk_path, args_line)
            ])

            convert_cmd = "bash -c \"{}\"".format(convert_cmd)

            self.debug_print("convert_cmd = \"{}\"".format(convert_cmd))

            result = Connection().shell(convert_cmd)
            self.debug_print(result)

            np.random.rand(*inputs[0].get_shape().as_list()).astype('float32').tofile(os.path.abspath(path + '.pb').replace('.pb', '.raw'))

            if not os.path.exists(os.path.abspath(path + '.pb').replace('.pb', '.dlc')):
                if self.debug_print is not print:
                    print('Fatal Error: Failed to generate required file.')
                    print("convert_cmd = \"{}\"".format(convert_cmd))
                    print(result)
                print('Generated DLC file:', os.path.abspath(path + '.pb').replace('.pb', '.dlc'))
                raise FileNotFoundError

    def _fetch_results(self,
                       connection: Connection, model_path,
                       input_size_list: List[List[int]], flags) -> InferenceResult:
        model_basename = os.path.basename(model_path)
        
        bench_cmd = '; '.join([
                'cd %s' % self.snpe_sdk_path,
                'source %s/bin/envsetup.sh -t %s' % (self.snpe_sdk_path, self.tensorflow_path),
                'rm sampledImg/*',
                'rm -r benchmark_res/results/*',
                'cp %s sampledImg/' % os.path.abspath(model_path + '.pb').replace('.pb', '.raw'),
                'echo "sampledImg/%s" > "sampledImg/imgList.txt"' % (model_basename + '.raw'),
                'cp benchmark_ref.json benchmarks/benchmark.json',
                'sed -i \'s#\[BENCHMARK_RUNTIMES\]#%s#g\' benchmarks/benchmark.json' % ', '.join(list(map(lambda x: '\\"' + x + '\\"', flags['runtimes']))),
                'sed -i \'s#\[INPUT_DLC_FILE\]#%s#g\' benchmarks/benchmark.json' % os.path.abspath(model_path + '.pb').replace('.pb', '.dlc'),
                'python benchmarks/snpe_bench.py -c benchmarks/benchmark.json -l detailed -p %s' % flags['performance_mode'],
                'cp benchmark_res/results/latest_results/benchmark_stats_BenchmarkPro.csv %s' % os.path.abspath(model_path + '.pb').replace('.pb', '_stats.csv'),
                'adb shell rm  /data/local/tmp/snpebm/BenchmarkPro/*.dlc',
                'adb shell rm  /data/local/tmp/snpebm/BenchmarkPro/sampledImg/*.raw'
        ])

        bench_cmd = "bash -c \"{}\"".format(bench_cmd)

        self.debug_print("bench_cmd = \"{}\"".format(bench_cmd))

        result = Connection().shell(bench_cmd)
        self.debug_print(result)

        model_stats = re.findall(self.bench_regex, open(os.path.abspath(model_path + '.pb').replace('.pb', '_stats.csv'), 'r').read())
        
        self.debug_print(model_stats)

        layerwise_info = []
        avg_stat = 0
        std_stat = 0

        for layer_info in model_stats:
            layerwise_info.append({
                        "name": '%s_%s_%s' % (layer_info[:3]),
                        "time": {
                            'avg_ms': float(layer_info[3]) / 1e3,
                            'std_ms': math.sqrt(((float(layer_info[4])-float(layer_info[3])) ** 2 + (float(layer_info[5])-float(layer_info[3])) ** 2) / 2) / 1e3
                            }
                    })
            avg_stat = avg_stat + float(layer_info[3]) / 1e3
            std_stat = std_stat + math.sqrt(((float(layer_info[4])-float(layer_info[3])) ** 2 + (float(layer_info[5])-float(layer_info[3])) ** 2) / 2) / 1e3
                
        self.debug_print('Layer Stat, Avg_ms: %.02fms, Std_ms: %.02fms' % (avg_stat, std_stat))

        return InferenceResult(avg_ms=std_stat, std_ms=std_stat, profiling_details=None, layerwise_info=layerwise_info)
