from .model_traverser import ModelTraverser, OpDetail

import inspect
import logging
from typing import List, Dict, Tuple, Optional
from functools import reduce
from collections import deque
import numpy as np

import tensorflow as tf


class ModelSplitter(ModelTraverser):
    def __init__(self, tflite_model_path: str):
        super().__init__(tflite_model_path)
        self.categories = {}
        self.all_layers = []

    def _op_to_category(self, op_detail: OpDetail) -> str:
        return "default"

    def _push_stack(self):
        frame = inspect.currentframe().f_back
        env = frame.f_locals
        code = frame.f_code
        varnames = list(code.co_varnames)[:code.co_argcount]
        varnames.remove("self")
        value = list(map(lambda x: env[x], varnames))
        key = self._op_to_category(env["op_detail"])
        if self.categories.get(key) is None:
            self.categories[key] = []
        self.categories[key].append(value)
        self.all_layers.append(value)

    @classmethod
    def _gen_activation_func(cls, name: str):
        from tensorflow.keras import activations
        if name == "RELU6":
            def ret(x):
                return activations.relu(x, max_value=6)
        elif name == "NONE":
            ret = None
        else:
            assert False
        return ret

    def conv_2d(
        self,
        op_detail: OpDetail,
        input_imsize: int, cin: int, cout: int,
        stride: int, ksize: int, activation: str
    ):
        self._push_stack()

    @classmethod
    def _construct_conv_2d(cls, nets: List[tf.Tensor], params) -> tf.Tensor:
        input_imsize, cin, cout, stride, ksize, activation = params[1:]
        return tf.keras.layers.Conv2D(
            filters=cout,
            kernel_size=[ksize, ksize],
            strides=[stride, stride],
            padding='same',
            activation=cls._gen_activation_func(activation)
        )(nets[0])

    def dwconv_2d(
        self,
        op_detail: OpDetail,
        input_imsize: int, cin: int,
        stride: int, ksize: int, activation: str
    ):
        self._push_stack()

    @classmethod
    def _construct_dwconv_2d(cls, nets: List[tf.Tensor], params) -> tf.Tensor:
        input_imsize, cin, stride, ksize, activation = params[1:]
        net = tf.nn.depthwise_conv2d(
            nets[0],
            filter=tf.constant(np.random.randn(
                ksize, ksize, cin, 1).astype(np.float32)),
            strides=[1, stride, stride, 1],
            padding='SAME',
            rate=[1, 1],
        )
        activation_func = cls._gen_activation_func(activation)
        if activation_func is not None:
            net = activation_func(net)
        return net

    def pool_2d(self, op_detail: OpDetail, input_imsize: int, cin: int, stride: int, ksize: int):
        self._push_stack()

    @classmethod
    def _construct_pool_2d(cls, nets: List[tf.Tensor], params) -> tf.Tensor:
        op_detail = params[0]
        input_imsize, cin, stride, ksize = params[1:]
        assert ksize == input_imsize

        args = [nets[0], [1, ksize, ksize, 1], [1, 1, 1, 1], 'VALID']
        if op_detail.type == "AVERAGE_POOL_2D":
            func = tf.nn.avg_pool
        elif op_detail.type == "MAX_POOL_2D":
            func = tf.nn.max_pool
        else:
            assert False

        return func(*args)

    def softmax(self, op_detail: OpDetail, num_inputs: int):
        self._push_stack()

    @classmethod
    def _construct_softmax(cls, nets: List[tf.Tensor], params) -> tf.Tensor:
        return tf.nn.softmax(nets[0])

    def add(self, op_detail: OpDetail, input_shapes: List[List[int]]):
        self._push_stack()

    @classmethod
    def _construct_add(cls, nets: List[tf.Tensor], params) -> tf.Tensor:
        assert len(nets) == 2
        return tf.math.add(nets[0], nets[1])

    def reshape(self, op_detail: OpDetail, from_shape: List[int], to_shape: List[int]):
        self._push_stack()

    @classmethod
    def _construct_reshape(cls, nets: List[tf.Tensor], params) -> tf.Tensor:
        from_shape, to_shape = params[1:]
        return tf.reshape(nets[0], to_shape)

    def construct_tf_graph(self, category_key: Optional[str]) \
            -> Tuple[List[tf.Tensor], List[tf.Tensor]]:
        if category_key is None:
            category = self.all_layers
        else:
            category = self.categories[category_key]

        tensors = set()
        for i in category:
            tensors.update(i[0].inputs)
            tensors.update(i[0].outputs)

        tensor_users = {key: [] for key in tensors}
        for i in range(len(category)):
            op_detail: OpDetail = category[i][0]
            for tensor in op_detail.inputs:
                tensor_users[tensor].append(i)

        node_refs = {
            idx: len(category[idx][0].inputs)
            for idx in range(len(category))
        }

        inputs_union = reduce(
            lambda acc, x: acc.union(x),
            map(lambda i: set(i[0].inputs), category),
            set()
        )
        outputs_union = reduce(
            lambda acc, x: acc.union(x),
            map(lambda i: set(i[0].outputs), category),
            set()
        )

        input_tensors = tensors.difference(outputs_union)
        output_tensors = tensors.difference(
            inputs_union).intersection(outputs_union)

        for tensor in input_tensors:
            for user_node in tensor_users[tensor]:
                node_refs[user_node] -= 1

        queue = [idx for idx in range(len(category)) if node_refs[idx] == 0]
        queue = deque(queue)

        tensor_pool: Dict[str, tf.Tensor] = {}

        placeholders = []

        def fetch_tf_tensor(name: str, loc) -> tf.Tensor:
            if tensor_pool.get(name) is None:
                tensor = self._fetch_tensor_with_name(name, *loc)
                tensor_pool[name] = tf.placeholder(
                    dtype=tf.float32,
                    shape=list(tensor.ShapeAsNumpy())
                )
                placeholders.append(tensor_pool[name])

            return tensor_pool[name]

        construct_func_params_dic = {
            "CONV_2D": {
                "inputs": [0],
                "func": self._construct_conv_2d
            },
            "DEPTHWISE_CONV_2D": {
                "inputs": [0],
                "func": self._construct_dwconv_2d
            },
            "AVERAGE_POOL_2D": {
                "inputs": [0],
                "func": self._construct_pool_2d
            },
            "MAX_POOL_2D": {
                "inputs": [0],
                "func": self._construct_pool_2d
            },
            "SOFTMAX": {
                "inputs": [0],
                "func": self._construct_softmax
            },
            "ADD": {
                "inputs": [0, 1],
                "func": self._construct_add
            },
            "RESHAPE": {
                "inputs": [0],
                "func": self._construct_reshape
            }
        }

        tf.reset_default_graph()

        while len(queue) >= 1:
            idx = queue.popleft()

            params = category[idx]
            op_detail: OpDetail = params[0]
            for tensor in op_detail.outputs:
                for user_node in tensor_users[tensor]:
                    node_refs[user_node] -= 1
                    if node_refs[user_node] == 0:
                        queue.append(user_node)

            assert op_detail.type in construct_func_params_dic
            construct_func_params = construct_func_params_dic[op_detail.type]
            nets = [
                fetch_tf_tensor(op_detail.inputs[idx], op_detail.loc)
                for idx in construct_func_params["inputs"]
            ]
            func = construct_func_params["func"]
            net = func(nets, params)
            tensor_pool[op_detail.outputs[0]] = net

        return (
            placeholders,
            list(map(lambda i: tensor_pool[i], output_tensors))
        )

    def split(self):
        self.categories = {}
        self.all_layers = []
        self.traverse()
