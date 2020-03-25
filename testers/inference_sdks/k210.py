import os
import re
import shutil

import serial

import math

from typing import List

import tensorflow as tf
import numpy as np

from .inference_sdk import InferenceSdk, InferenceResult
from utils.utils import concatenate_flags
from .utils import rfind_assign_float
from utils.connection import Connection
from utils.tf_model_utils import to_saved_model

class K210(InferenceSdk):
    @staticmethod
    def default_settings():
        return {
            **InferenceSdk.default_settings(),
            'k210_sdk_path': '/home/hanxiao/benchmark_platform/kendryte-standalone-sdk',  
            'k210_toochain_path': '/home/hanxiao/benchmark_platform/kendryte-toolchain',
            'k210_nncase_path': '/home/hanxiao/benchmark_platform/ncc' ,
            'k210_usb_port': '/dev/ttyUSB0',      
            'debug_print': False,
        }
    
    @staticmethod
    def default_flags():
        return {
            **InferenceSdk.default_flags(),
        }

    def __init__(self, settings={}):
        super().__init__(settings)
        self.k210_sdk_path = self.settings["k210_sdk_path"]
        self.k210_toochain_path = self.settings["k210_toochain_path"]
        self.k210_nncase_path = self.settings["k210_nncase_path"]
        self.k210_usb_port = self.settings["k210_usb_port"]
        self.debug_print = print if self.settings["debug_print"] == True else lambda x: x


    def generate_model(self, path, inputs, outputs):
        path = os.path.splitext(path)[0]
        model_basename = os.path.basename(path)

        outputs_ops_names = [o.op.name for o in outputs]

        if os.path.exists(os.path.join(path, 'model')):
            shutil.rmtree(os.path.join(path, 'model'))
        
        try:
            os.remove('model.kmodel')
        except:
            pass
    
        try:
            os.remove('model.tflite')
        except:
            pass

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            to_saved_model(
                sess, inputs, outputs, path,
                replace_original_dir=True
            )

            assert 1 == len(inputs)
            assert 1 == len(outputs_ops_names)
            
            if os.path.exists(path + '.dir'):
                shutil.rmtree(path + '.dir')

            os.mkdir(path + '.dir')

            converter = tf.lite.TFLiteConverter.from_saved_model(path)
            model = converter.convert()

            with tf.gfile.FastGFile(path + '.tflite', mode='wb') as f:
                f.write(model)

            input_img = np.random.randint(size=inputs[0].get_shape().as_list(), low=0, high=254)
            input_img.astype(np.float32).tofile(os.path.join(os.path.abspath(path + '.dir'), 'input_img.raw'))

            with tf.gfile.FastGFile(os.path.join(os.path.abspath(path + '.dir'), 'input_img.c'), mode='w+') as f:
                f.write('const unsigned char gImage_image[]  __attribute__((aligned(128))) = {\n')
                f.write(', '.join([str(i) for i in input_img.flatten()]))
                f.write('\n};')

            args = [
                'compile',
                '%s' % os.path.abspath(path + '.tflite'),
                '%s' % os.path.abspath(path + '.tflite').replace('.tflite', '.kmodel'),
                '-i tflite', 
                '--dataset %s' % os.path.abspath(path + '.dir'),
                '--dataset-format raw',
            ]

            args_line = ' '.join(args)

            convert_cmd = '; '.join([
                    
                    'rm %s' % (os.path.join(self.k210_sdk_path, 
                                'build', 'CMakeFiles', 'benchmarks.dir', 'src', 'benchmarks', '*')),

                    'mv %s %s'% (os.path.join(os.path.abspath(path + '.dir'), 'input_img.c')
                              , os.path.join(self.k210_sdk_path, 'src', 'benchmarks')),

                    '%s %s' % (self.k210_nncase_path, args_line),

                    'cp %s %s' % (os.path.abspath(path + '.tflite').replace('.tflite', '.kmodel')
                                , os.path.join(self.k210_sdk_path, 'src', 'benchmarks')),

                    'cd %s' % (os.path.join(self.k210_sdk_path, 'build')),
                    
                    'make -j',
                    
                    'python3 -mkflash -b1500000 -p%s -BgoB %s' % (self.k210_usb_port, 'benchmarks.bin')
            ])

            map(str,inputs[0].get_shape().as_list())

            convert_cmd = "bash -c \"{}\"".format(convert_cmd)

            self.debug_print("convert_cmd = \"{}\"".format(convert_cmd))

            result = Connection().shell(convert_cmd)
            self.debug_print(result)

            if not os.path.exists(os.path.abspath(path + '.tflite').replace('.tflite', '.kmodel')):
                if self.debug_print is not print:
                    print('Fatal Error: Failed to generate required file.')
                    print("convert_cmd = \"{}\"".format(convert_cmd))
                    print(result)
                print('Generated DLC file:', os.path.abspath(path + '.pb').replace('.pb', '.dlc'))
                raise FileNotFoundError

    def _fetch_results(self,
                       connection: Connection, model_path,
                       input_size_list: List[List[int]], flags) -> InferenceResult:

        model_path = os.path.splitext(model_path)[0]
        model_basename = os.path.basename(model_path)
        
        layerwise_info = []
        avg_stat = 0.0
    
        with serial.Serial(self.k210_usb_port, 115200, timeout=10) as ser:
            while 1:
                try:
                    line = ser.readline().decode('utf-8')
                    if line == '':
                        raise FileNotFoundError
                    if '######BENCHMARK_START######' in line:
                        pass
                    if '######BENCHMARK_END######' in line:
                        break
                    if ': ' in line and 'Total' not in line:
                        layerwise_info.append({
                                "name": line.split(':')[0],
                                "time": {
                                    'avg_ms': float(re.findall(r'[\d|.]{2,}', line)[0]),
                                    'std_ms': 0
                                    }
                            })
                    if 'Total' in line:
                        avg_stat = float(re.findall(r'[\d|.]{2,}', line)[0])
                        break
                except:
                    pass
        
        self.debug_print(layerwise_info)
                
        self.debug_print('Layer Stat, Avg_ms: %.02fms' % (avg_stat))

        return InferenceResult(avg_ms=avg_stat, std_ms=0.0, profiling_details=None, layerwise_info=layerwise_info)
