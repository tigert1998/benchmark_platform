from testers.tester import Tester

import tensorflow as tf

from typing import Tuple, List


class TestSingleLayer(Tester):
    @staticmethod
    def _pad_before_input(shapes: List[List[int]]) -> Tuple[List[tf.Tensor], List[tf.Tensor]]:
        """Generate a network before the formal network to benchmark,
        since the padded network may help eliminate some unnecessary overhead.
        Args:
            shapes: the shapes of output_tensors
        Returns:
            input_tensors: the inputs of the padded network
            output_tensors: the outputs of the padded network and the inputs of the formal network
        """
        tensors = [
            tf.placeholder(
                name="input_im_{}".format(i),
                dtype=tf.float32,
                shape=shape
            ) for i, shape in zip(range(len(shapes)), shapes)
        ]
        return tensors, tensors

    @staticmethod
    def _pad_after_output(output_tensors: List[tf.Tensor]) -> List[tf.Tensor]:
        """Generate a network after the formal network to benchmark.
        Args:
            output_tensors: the outputs of the formal network
        Returns:
            outputs of the padded network after the formal network
        """
        return output_tensors

    def _generate_tf_model(self, sample) -> Tuple[List[tf.Tensor], List[tf.Tensor]]:
        ...

    def _generate_model(self, sample, model_path: str) -> List[List[int]]:
        """According to the sample, generate a single-layer model with padded networks,
        whose format is decided by inference SDK. This method invokes self._generate_tf_model
        to get input tensors and output tensors.

        Args:
            model_path: model path without extension
        Returns:
            a list of input sizes
        """
        tf.reset_default_graph()
        inputs, outputs = self._generate_tf_model(sample)
        self.inference_sdk.generate_model(model_path, inputs, outputs)
        return list(map(lambda tensor: tensor.get_shape().as_list(), inputs))

    def _test_sample(self, sample):
        model_path = "model"
        input_size_list = self._generate_model(sample, model_path)
        return self.inference_sdk.fetch_results(
            self.connection, model_path, input_size_list, self.benchmark_model_flags)
