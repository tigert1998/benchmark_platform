import os
import sys
import logging

from .tflite.Conv2DOptions import Conv2DOptions
from .tflite.ReshapeOptions import ReshapeOptions
from .tflite.SoftmaxOptions import SoftmaxOptions
from .tflite.Pool2DOptions import Pool2DOptions
from .tflite.DepthwiseConv2DOptions import DepthwiseConv2DOptions

from .tflite.SubGraph import SubGraph
from .tflite.Model import Model
from .tflite.BuiltinOptions import BuiltinOptions
from .tflite.Operator import Operator


def BuiltinOptionsType_to_str(code: BuiltinOptions) -> str:
    for name, value in BuiltinOptions.__dict__.items():
        if value == code:
            return name
    assert False


class TfliteModelManager:
    def __init__(self, tflite_model_path: str):
        assert tflite_model_path.endswith(".tflite")
        buf = open(tflite_model_path, 'rb').read()
        self.model = Model.GetRootAsModel(buf, 0)

    def conv_2d(self, input_imsize: int, cin: int, cout: int, stride: int, ksize: int):
        msg = f"Unimplemented Conv: {[input_imsize, cin, cout, stride, ksize]}"
        logging.warn(msg)

    def dwconv_2d(self, input_imsize: int, cin: int, stride: int, ksize: int):
        msg = f"Unimplemented DWConv: {[input_imsize, cin, stride, ksize]}"
        logging.warn(msg)

    def pool_2d(self, input_imsize: int, cin: int, stride: int, ksize: int):
        msg = f"Unimplemented Pool: {[input_imsize, cin, stride, ksize]}"
        logging.warn(msg)

    def softmax(self, num_inputs: int):
        msg = f"Unimplemented Softmax: {[1, num_inputs]}"
        logging.warn(msg)

    def _pool_2d(self, subgraph: SubGraph, op: Operator):
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

        self.pool_2d(input_imsize, cin, stride, ksize)

    def _conv_2d(self, subgraph, op):
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

        self.conv_2d(input_imsize, cin, cout, stride, ksize)

    def _dwconv_2d(self, subgraph, op):
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

        self.dwconv_2d(input_imsize, cin, stride, ksize)

    def _softmax(self, subgraph, op):
        builtin_options = op.BuiltinOptions()
        options = SoftmaxOptions()
        options.Init(builtin_options.Bytes, builtin_options.Pos)

        assert op.InputsLength() == 1 and op.OutputsLength() == 1
        input_tensor = subgraph.Tensors(op.Inputs(0))
        assert input_tensor.ShapeLength() == 2 and input_tensor.Shape(0) == 1

        num_inputs = input_tensor.Shape(1)
        self.softmax(num_inputs)

    def traverse(self):
        assert 1 == self.model.SubgraphsLength()
        subgraph = self.model.Subgraphs(0)

        for i in range(subgraph.OperatorsLength()):
            op = subgraph.Operators(i)
            builtin_option_type = op.BuiltinOptionsType()

            args = [subgraph, op]
            if builtin_option_type == BuiltinOptions.Conv2DOptions:
                self._conv_2d(*args)
            elif builtin_option_type == BuiltinOptions.DepthwiseConv2DOptions:
                self._dwconv_2d(*args)
            elif builtin_option_type == BuiltinOptions.Pool2DOptions:
                self._pool_2d(*args)
            elif builtin_option_type == BuiltinOptions.SoftmaxOptions:
                self._softmax(*args)
            else:
                msg = f"Ignored operator: {BuiltinOptionsType_to_str(builtin_option_type)}"
                logging.warn(msg)
