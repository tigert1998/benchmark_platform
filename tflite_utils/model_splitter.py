from .model_traverser import ModelTraverser, OpDetail

import inspect
from typing import List
from functools import reduce
from collections import deque

import tensorflow as tf


class ModelSplitter(ModelTraverser):
    def __init__(self, tflite_model_path: str):
        super().__init__(tflite_model_path)
        self.categories = {}

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

    def conv_2d(
        self,
        op_detail: OpDetail,
        input_imsize: int, cin: int, cout: int,
        stride: int, ksize: int, activation: str
    ):
        self._push_stack()

    def dwconv_2d(
        self,
        op_detail: OpDetail,
        input_imsize: int, cin: int,
        stride: int, ksize: int, activation: str
    ):
        self._push_stack()

    def pool_2d(self, op_detail: OpDetail, input_imsize: int, cin: int, stride: int, ksize: int):
        self._push_stack()

    def softmax(self, op_detail: OpDetail, num_inputs: int):
        self._push_stack()

    def add(self, op_detail: OpDetail, input_shapes: List[List[int]]):
        self._push_stack()

    def reshape(self, op_detail: OpDetail, from_shape: List[int], to_shape: List[int]):
        self._push_stack()

    def construct_tf_graph(self, category_key: str):
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

        input_tensors = tensors.difference(reduce(
            lambda acc, x: acc.union(x),
            map(lambda i: set(i[0].outputs), category)),
            set()
        )

        for tensor in input_tensors:
            for user_node in tensor_users[tensor]:
                node_refs[user_node] -= 1

        queue = [idx for idx in range(len(category)) if node_refs[idx] == 0]
        queue = deque(queue)

        tf.reset_default_graph()

        while len(queue) >= 1:
            idx = queue.popleft()

            arr = category[idx]
            op_detail: OpDetail = arr[0]
            for tensor in op_detail.outputs:
                for user_node in tensor_users[tensor]:
                    node_refs[user_node] -= 1
                    if node_refs[user_node] == 0:
                        queue.append(user_node)

            print(op_detail.type)

    def split(self):
        self.traverse()
