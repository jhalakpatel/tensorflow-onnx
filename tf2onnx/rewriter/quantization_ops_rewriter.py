# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

"""
tf2onnx.rewriter - rewrite tensorflow FakeQuantMinMaxVars op
"""

from tf2onnx.graph_matcher import OpTypePattern, GraphMatcher

import numpy as np
import struct


from tf2onnx.handler import tf_op
from tf2onnx import utils
from tf2onnx import constants

# pylint: disable=missing-docstring

def extract_numpy_array(node):
    return np.frombuffer(node.attr["value"].t.raw_data, dtype="float32")

def convertRSCKtoKCRS(weight):
    val = weight.get_tensor_value(as_list=False)
    if len(weight.output_shapes[0]) == 2:
       shape = weight.output_shapes[0]
       val = np.reshape(val, (shape[0], shape[1], 1, 1))
       val = val.transpose(constants.HWCN_TO_NCHW)
    else:
        val = val.transpose(constants.HWCN_TO_NCHW)
    weight.set_tensor_value(val)

def create_qdq_nodes(g, match_results):
    for match in match_results:
        old_output = match.get_op('output')
        output_dtype = g.get_dtype(old_output.output[0])
        output_shape = g.get_shape(old_output.output[0])

        min_value = extract_numpy_array(old_output.inputs[1])
        max_value = extract_numpy_array(old_output.inputs[2])
        dynamic_range = max(abs(min_value), abs(max_value))
        quant_scale = dynamic_range / 127.0
        dequant_scale = dynamic_range / 127.0

        y_quant_scale = g.make_const(name=utils.make_name("y_quant_scale"), np_val = quant_scale)
        y_zero_point = g.make_const(name=utils.make_name("y_zero_point"), np_val=np.int8(0))
        quant_node = g.make_node(op_type = "QuantizeLinear", inputs=[old_output.input[0], y_quant_scale.output[0], y_zero_point.output[0]], shapes=[output_shape], dtypes=[output_dtype], name=utils.make_name("QuantLinearNode"))
        g.set_shape(quant_node.output[0], output_shape)

        if quant_node.inputs[0].is_const():
            convertRSCKtoKCRS(quant_node.inputs[0])

        g.remove_node(old_output.name)

        y_dequant_scale = g.make_const(name=utils.make_name("y_dequant_scale"), np_val = dequant_scale)
        y_inv_zero_point = g.make_const(name=utils.make_name("y_inv_zero_point"), np_val=np.int8(0))
        dequant_node = g.make_node(op_type = "DequantizeLinear", inputs=[quant_node.output[0], y_dequant_scale.output[0], y_inv_zero_point.output[0]], outputs = [old_output.output[0]], shapes=[output_shape], dtypes=[output_dtype], name=utils.make_name("DequantLinearNode"))
        g.set_shape(dequant_node.output[0], output_shape)
    return g.get_nodes()

def rewrite_quantize_and_dequantize_v3(g, ops):
    pattern = \
        OpTypePattern('QuantizeAndDequantizeV3', name='output', inputs=[
            OpTypePattern("*"),
            OpTypePattern(None),
            OpTypePattern(None),
            OpTypePattern(None),
        ])
    matcher = GraphMatcher(pattern)
    match_results = list(matcher.match_ops(ops))
    return create_qdq_nodes(g, match_results)

def rewrite_fake_quant_with_min_max_vars(g, ops):
    pattern = \
        OpTypePattern('FakeQuantWithMinMaxVars', name='output', inputs=[
            OpTypePattern("*"),
            OpTypePattern(None),
            OpTypePattern(None),
        ])
    matcher = GraphMatcher(pattern)
    match_results = list(matcher.match_ops(ops))
    return create_qdq_nodes(g, match_results)