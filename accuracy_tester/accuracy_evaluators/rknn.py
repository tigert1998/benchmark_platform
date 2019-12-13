from .accuracy_evaluator_def import AccuracyEvaluatorDef

from rknn.api import RKNN

import cv2
import numpy as np


def show_outputs(outputs):
    output = outputs[0][0]
    output_sorted = sorted(output, reverse=True)
    top5_str = 'mobilenet_v2\n-----TOP 5-----\n'
    for i in range(5):
        value = output_sorted[i]
        index = np.where(output == value)
        for j in range(len(index)):
            if (i + j) >= 5:
                break
            if value > 0:
                topi = '{}: {}\n'.format(index[j], value)
            else:
                topi = '-1: 0.0\n'
            top5_str += topi
    print(top5_str)


class Rknn(AccuracyEvaluatorDef):
    def __init__(self, settings):
        super().__init__(settings)
        self.rknn = RKNN()

    def evaluate_models(self, model_paths, image_path_label_gen):
        # pre-process config
        # print('--> config model')
        for x, y in image_path_label_gen:
            print(x, y)

        # self.rknn.config(
        #     channel_mean_value='103.94 116.78 123.68 58.82',
        #     reorder_channel='2 1 0'
        # )

        # # Load tensorflow model
        # print('--> Loading model')
        # assert 0 == self.rknn.load_caffe(
        #     model='./mobilenet_v2.prototxt',
        #     proto='caffe',
        #     blobs='./mobilenet_v2.caffemodel')

        # # Build model
        # print('--> Building model')
        # assert 0 == self.rknn.build(do_quantization=False)

        # # Export rknn model
        # print('--> Export RKNN model')
        # assert 0 == self.rknn.export_rknn('./mobilenet_v2.rknn')

        # # Set inputs
        # img = cv2.imread('./goldfish_224x224.jpg')
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # print('--> Init runtime environment')
        # assert 0 == self.rknn.init_runtime(target="rk1808")

        # # Inference
        # print('--> Running model')
        # outputs = self.rknn.inference(inputs=[img])
        # show_outputs(outputs)

        # # perf
        # print('--> Begin evaluate model performance')
        # perf_results = self.rknn.eval_perf(inputs=[img])
