import os
import sys
import logging
from typing import List, Tuple
from collections import namedtuple

from .tflite.Model import Model
from .tflite.SubGraph import SubGraph
from .tflite.Operator import Operator

from .tflite.BuiltinOperator import BuiltinOperator
from .tflite.ActivationFunctionType import ActivationFunctionType


def fbs_enum_to_str(enum_type):
    def func(code: int) -> str:
        for name, value in enum_type.__dict__.items():
            if value == code:
                return name
        assert False
    return func


BuiltinOperator_to_str = fbs_enum_to_str(BuiltinOperator)
ActivationFunctionType_to_str = fbs_enum_to_str(ActivationFunctionType)

OpDetail = namedtuple("OpDetail", [
    "loc", "inputs", "outputs"
])


class ModelTraverser:
    def __init__(self, tflite_model_path: str):
        assert tflite_model_path.endswith(".tflite")
        buf = open(tflite_model_path, 'rb').read()
        self.model = Model.GetRootAsModel(buf, 0)

    def _fetch_subgraph_and_op(self, subgraph_idx: int, op_idx: int) -> Tuple[SubGraph, Operator]:
        subgraph = self.model.Subgraphs(subgraph_idx)
        op = subgraph.Operators(op_idx)
        return subgraph, op

    def _construct_op_detail(self, subgraph_idx: int, op_idx: int) -> OpDetail:
        subgraph, op = self._fetch_subgraph_and_op(subgraph_idx, op_idx)
        inputs = [subgraph.Tensors(i).Name().decode('utf-8')
                  for i in op.InputsAsNumpy()]
        outputs = [subgraph.Tensors(i).Name().decode('utf-8')
                   for i in op.OutputsAsNumpy()]
        return OpDetail(
            loc=(subgraph_idx, op_idx),
            inputs=inputs,
            outputs=outputs
        )

    def conv_2d(
        self,
        op_detail: OpDetail,
        input_imsize: int, cin: int, cout: int,
        stride: int, ksize: int, activation: str
    ):
        msg = f"Unimplemented Conv: {[input_imsize, cin, cout, stride, ksize, activation]}"
        logging.warn(msg)

    def dwconv_2d(
        self,
        op_detail: OpDetail,
        input_imsize: int, cin: int,
        stride: int, ksize: int, activation: str
    ):
        msg = f"Unimplemented DWConv: {[input_imsize, cin, stride, ksize, activation]}"
        logging.warn(msg)

    def pool_2d(self, op_detail: OpDetail, input_imsize: int, cin: int, stride: int, ksize: int):
        msg = f"Unimplemented Pool: {[input_imsize, cin, stride, ksize]}"
        logging.warn(msg)

    def softmax(self, op_detail: OpDetail, num_inputs: int):
        msg = f"Unimplemented Softmax: {[1, num_inputs]}"
        logging.warn(msg)

    def add(self, op_detail: OpDetail, input_shapes: List[List[int]]):
        msg = f"Unimplemented Add: {input_shapes}"
        logging.warn(msg)

    def _pool_2d(self, subgraph_idx, op_idx):
        from .tflite.Pool2DOptions import Pool2DOptions

        subgraph, op = self._fetch_subgraph_and_op(subgraph_idx, op_idx)
        op_detail = self._construct_op_detail(subgraph_idx, op_idx)
        builtin_options = op.BuiltinOptions()
        options = Pool2DOptions()
        options.Init(builtin_options.Bytes, builtin_options.Pos)

        assert options.StrideH() == options.StrideW()
        stride = options.StrideH()

        assert op.InputsLength() == 1 and op.OutputsLength() == 1
        input_tensor = subgraph.Tensors(op.Inputs(0))
        output_tensor = subgraph.Tensors(op.Outputs(0))
        cin = input_tensor.Shape(3)
        input_imsize = input_tensor.Shape(1)

        assert options.FilterHeight() == options.FilterWidth()
        ksize = options.FilterHeight()

        self.pool_2d(op_detail, input_imsize, cin, stride, ksize)

    def _conv_2d(self, subgraph_idx, op_idx):
        from .tflite.Conv2DOptions import Conv2DOptions

        subgraph, op = self._fetch_subgraph_and_op(subgraph_idx, op_idx)
        op_detail = self._construct_op_detail(subgraph_idx, op_idx)
        builtin_options = op.BuiltinOptions()
        options = Conv2DOptions()
        options.Init(builtin_options.Bytes, builtin_options.Pos)

        assert options.DilationHFactor() == 1 and options.DilationWFactor() == 1
        assert options.StrideH() == options.StrideW()
        stride = options.StrideH()

        assert op.InputsLength() == 3 and op.OutputsLength() == 1
        input_tensor = subgraph.Tensors(op.Inputs(0))
        output_tensor = subgraph.Tensors(op.Outputs(0))
        cin = input_tensor.Shape(3)
        cout = output_tensor.Shape(3)

        assert input_tensor.Shape(1) == input_tensor.Shape(2)
        input_imsize = input_tensor.Shape(1)

        weight_tensor = subgraph.Tensors(op.Inputs(1))
        assert weight_tensor.Shape(1) == weight_tensor.Shape(2)
        ksize = weight_tensor.Shape(1)

        activation = ActivationFunctionType_to_str(
            options.FusedActivationFunction()
        )
        self.conv_2d(op_detail, input_imsize, cin,
                     cout, stride, ksize, activation)

    def _dwconv_2d(self, subgraph_idx, op_idx):
        from .tflite.DepthwiseConv2DOptions import DepthwiseConv2DOptions

        subgraph, op = self._fetch_subgraph_and_op(subgraph_idx, op_idx)
        op_detail = self._construct_op_detail(subgraph_idx, op_idx)
        builtin_options = op.BuiltinOptions()
        options = DepthwiseConv2DOptions()
        options.Init(builtin_options.Bytes, builtin_options.Pos)

        assert options.DilationHFactor() == 1 and options.DilationWFactor() == 1
        assert options.StrideH() == options.StrideW()
        stride = options.StrideH()

        assert op.InputsLength() == 3 and op.OutputsLength() == 1
        input_tensor = subgraph.Tensors(op.Inputs(0))
        output_tensor = subgraph.Tensors(op.Outputs(0))
        cin = input_tensor.Shape(3)
        cout = output_tensor.Shape(3)
        assert cin == cout

        assert input_tensor.Shape(1) == input_tensor.Shape(2)
        input_imsize = input_tensor.Shape(1)

        weight_tensor = subgraph.Tensors(op.Inputs(1))
        assert weight_tensor.Shape(1) == weight_tensor.Shape(2)
        ksize = weight_tensor.Shape(1)

        activation = ActivationFunctionType_to_str(
            options.FusedActivationFunction()
        )
        self.dwconv_2d(op_detail, input_imsize, cin, stride, ksize, activation)

    def _softmax(self, subgraph_idx, op_idx):
        from .tflite.SoftmaxOptions import SoftmaxOptions

        subgraph, op = self._fetch_subgraph_and_op(subgraph_idx, op_idx)
        op_detail = self._construct_op_detail(subgraph_idx, op_idx)
        builtin_options = op.BuiltinOptions()
        options = SoftmaxOptions()
        options.Init(builtin_options.Bytes, builtin_options.Pos)

        assert op.InputsLength() == 1 and op.OutputsLength() == 1
        input_tensor = subgraph.Tensors(op.Inputs(0))
        assert input_tensor.ShapeLength() == 2 and input_tensor.Shape(0) == 1

        num_inputs = input_tensor.Shape(1)
        self.softmax(op_detail, num_inputs)

    def _add(self, subgraph_idx, op_idx):
        from .tflite.AddOptions import AddOptions

        subgraph, op = self._fetch_subgraph_and_op(subgraph_idx, op_idx)
        op_detail = self._construct_op_detail(subgraph_idx, op_idx)
        builtin_options = op.BuiltinOptions()
        options = AddOptions()
        options.Init(builtin_options.Bytes, builtin_options.Pos)

        assert op.OutputsLength() == 1

        input_tensors = [
            subgraph.Tensors(op.Inputs(i))
            for i in range(op.InputsLength())
        ]
        input_shapes = list(map(
            lambda tensor: list(tensor.ShapeAsNumpy()),
            input_tensors))
        self.add(op_detail, input_shapes)

    def traverse(self):
        assert 1 == self.model.SubgraphsLength()
        subgraph = self.model.Subgraphs(0)

        for i in range(subgraph.OperatorsLength()):
            op = subgraph.Operators(i)
            opcode = self.model.OperatorCodes(op.OpcodeIndex())
            assert opcode.CustomCode() is None

            builtin_operator = opcode.BuiltinCode()

            args = [0, i]
            if builtin_operator == BuiltinOperator.CONV_2D:
                self._conv_2d(*args)
            elif builtin_operator == BuiltinOperator.DEPTHWISE_CONV_2D:
                self._dwconv_2d(*args)
            elif builtin_operator in [BuiltinOperator.AVERAGE_POOL_2D, BuiltinOperator.MAX_POOL_2D]:
                self._pool_2d(*args)
            elif builtin_operator == BuiltinOperator.SOFTMAX:
                self._softmax(*args)
            elif builtin_operator == BuiltinOperator.ADD:
                self._add(*args)
            else:
                msg = f"Ignored operator: {BuiltinOperator_to_str(builtin_operator)}"
                logging.warn(msg)
