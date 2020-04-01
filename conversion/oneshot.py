import os

from .to_tflite import *
from utils.tf_model_utils import to_saved_model, load_graph, analyze_inputs_outputs
from preprocess.model_archive import ModelDetail


def oneshot(
    model_detail: ModelDetail,
    validation_folder: str
):
    assert tf.__version__.startswith("2.1.")
    assert model_detail.model_path.endswith(".pb")
    model_name = os.path.basename(model_detail.model_path[:-3])

    # to saved model
    graph = load_graph(model_detail.model_path)
    with tf.compat.v1.Session(graph=graph) as sess:
        inputs, outputs = analyze_inputs_outputs(graph)
        to_saved_model(sess, inputs, outputs, model_name, False)

    # to tflite
    to_tflite_params = {
        no_quant: [],
        int_quant: [
            None, {
                "validation_folder": validation_folder,
                "preprocess": model_detail.preprocess
            }
        ],
        weight_quant: [],
        float16_quant: []
    }

    for method, params in to_tflite_params.items():
        converter = tf.lite.TFLiteConverter.from_saved_model(model_name)
        method(converter, *params)
        model = converter.convert()
        open("{}_{}.tflite".format(model_name, method.__name__), "wb").write(model)
