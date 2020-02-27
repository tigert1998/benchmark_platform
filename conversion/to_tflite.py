import tensorflow as tf
import argparse
import numpy as np

from typing import Optional, List


def get_inputs_shapes(saved_model_path):
    from tensorflow.python.saved_model import tag_constants
    with tf.compat.v1.Session(graph=tf.Graph()) as sess:
        imported = tf.compat.v1.saved_model.load(
            sess, [tag_constants.SERVING], saved_model_path)
        inputs = imported.signature_def["serving_default"].inputs
        ans = []
        for _, value in inputs.items():
            dim = value.tensor_shape.dim
            ans.append([dim[i].size for i in range(len(dim))])
        return ans


def int_quant(
    converter: tf.lite.TFLiteConverter,
    inputs_shapes: Optional[List[List[int]]] = None,
        imagenet_model_config=None):

    if imagenet_model_config is None:
        def representative_data_gen():
            for _ in range(3):
                yield list(map(lambda shape: np.random.randn(*shape).astype(np.float32), inputs_shapes))
    else:
        validation_folder = imagenet_model_config["validation_folder"]
        preprocess = imagenet_model_config["preprocess"]

        def representative_data_gen():
            for image_path in glob("{}/*.JPEG".format(validation_folder))[:1000]:
                yield [preprocess.execute(image_path)]

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8


def weight_quant(converter: tf.lite.TFLiteConverter):
    converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]


def float16_quant(converter: tf.lite.TFLiteConverter):
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]


def no_quant(converter: tf.lite.TFLiteConverter):
    converter.optimizations = []


def main(saved_model_path: str, quantization: str, output_path: Optional[str]):
    print("saved_model_path = {}\nquantization = {}".format(
        saved_model_path, quantization))

    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)

    if quantization == "":
        no_quant(converter)
    elif quantization == "int":
        int_quant(converter, get_inputs_shapes(saved_model_path))
    elif quantization == "weight":
        weight_quant(converter)
    elif quantization == "float16":
        float16_quant(converter)
    else:
        assert False

    model = converter.convert()
    if output_path is None:
        output_path = saved_model_path + ".tflite"
    open(output_path, "wb").write(model)


if __name__ == "__main__":
    assert tf.__version__.startswith("2.1.")
    parser = argparse.ArgumentParser(
        description='convert saved_model to tflite')
    parser.add_argument('--saved_model_path', type=str, required=True)
    parser.add_argument('--quantization', type=str, choices=[
                        "", "weight", "int", "float16"], required=True)
    parser.add_argument('--output_path', type=str)
    args = parser.parse_args()

    get_inputs_shapes(args.saved_model_path)

    main(args.saved_model_path, args.quantization, args.output_path)
