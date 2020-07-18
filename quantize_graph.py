# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Transforms a float-trained graph into an equivalent quantized version.

An example of command-line usage is:
bazel build tensorflow/tools/quantization:quantize_graph \
&& bazel-bin/tensorflow/tools/quantization/quantize_graph \
--input=tensorflow_inception_graph.pb
--output_node_names="softmax2" --print_nodes --output=/tmp/quantized_graph.pb \
--mode=eightbit --logtostderr

To quantize for Intel CPU, add --intel_cpu_eightbitize=True.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import re
import numpy as np

from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import app
from tensorflow.python.platform import flags as flags_lib
from tensorflow.python.platform import gfile
from google.protobuf import text_format

strip_redundant_quantization = True

def print_input_nodes(current_node, nodes_map, indent, already_visited):
    print(" " * indent + current_node.op + ":" + current_node.name)
    already_visited[current_node.name] = True
    for input_node_name in current_node.input:
       if input_node_name in already_visited:
        continue
       input_node = nodes_map[input_node_name]
       print_input_nodes(input_node, nodes_map, indent + 1, already_visited)


def create_node(op, name, inputs):
    new_node = node_def_pb2.NodeDef()
    new_node.op = op
    new_node.name = name
    for input_name in inputs:
        if input_node_name in already_visited:
         continue
        input_node = nodes_map[input_node_name]
        print_input_nodes(input_node, nodes_map, indent + 1, already_visited)


def create_node(op, name, inputs):
    new_node = node_def_pb2.NodeDef()
    new_node.op = op
    new_node.name = name
    for input_name in inputs:
        new_node.input.extend([input_name])
    return new_node


def create_constant_node(name, value, dtype, shape=None):
    node = create_node("Const", name, [])
    set_attr_dtype(node, "dtype", dtype)
    set_attr_tensor(node, "value", value, dtype, shape)
    return node


def copy_attr(node, key, attr_value):
    try:
        node.attr[key].CopyFrom(attr_value)
    except KeyError:
        pass


def set_attr_dtype(node, key, value):
    try:
        node.attr[key].CopyFrom(
            attr_value_pb2.AttrValue(type=value.as_datatype_enum))
    except KeyError:
        pass


def set_attr_shape(node, key, value):
    try:
        node.attr[key].CopyFrom(
            attr_value_pb2.AttrValue(shape=tensor_shape.as_shape(value).as_proto()))
    except KeyError:
        pass


def set_attr_tensor(node, key, value, dtype, shape=None):
    try:
        node.attr[key].CopyFrom(
            attr_value_pb2.AttrValue(tensor=tensor_util.make_tensor_proto(
                value, dtype=dtype, shape=shape)))
    except KeyError:
        pass


def set_attr_string(node, key, value):
    try:
        node.attr[key].CopyFrom(attr_value_pb2.AttrValue(s=value))
    except KeyError:
        pass


def set_attr_int_list(node, key, value):
    list_value = attr_value_pb2.AttrValue.ListValue(i=value)
    try:
        node.attr[key].CopyFrom(attr_value_pb2.AttrValue(list=list_value))
    except KeyError:
        pass

def set_attr_int_list(node, key, value):
    list_value = attr_value_pb2.AttrValue.ListValue(i=value)
    try:
        node.attr[key].CopyFrom(attr_value_pb2.AttrValue(list=list_value))
    except KeyError:
        pass

def set_attr_bool(node, key, value):
    try:
        node.attr[key].CopyFrom(attr_value_pb2.AttrValue(b=value))
    except KeyError:
        pass


def set_attr_int(node, key, value):
    try:
        node.attr[key].CopyFrom(attr_value_pb2.AttrValue(i=value))
    except KeyError:
        pass


def set_attr_float(node, key, value):
    try:
        node.attr[key].CopyFrom(attr_value_pb2.AttrValue(f=value))
    except KeyError:
        pass


def node_name_from_input(node_name):
    # print(node_name)
    """Strips off ports and other decorations to get the underlying node name."""
    if node_name.startswith("^"):
        node_name = node_name[1:]
    m = re.search(r"(.*):\d+$", node_name)
    if m:
        node_name = m.group(1)
    # print(node_name)
    return node_name


def ensure_tensor_name_has_port(node_name):
    """Makes sure that a tensor name has :0 if no explicit port exists."""
    m = re.search(r"(.*):\d+$", node_name)
    if m:
        name_with_port = node_name
    else:
        name_with_port = node_name + ":0"
    return name_with_port


def unique_node_name_from_input(node_name):
    """Replaces invalid characters in input names to get a unique node name."""
    return node_name.split("/")[-1]
    # return node_name.replace(":", "__port__").replace("^", "__hat__")
# TODO(intel-tf): Current Intel-CPU quantized Conv2D and Matmul supports only
# signed scaled mode of weight quantization.

def quantize_weight_eightbit(input_node, quantization_mode):
    
    """Returns replacement nodes for input_node using the Dequantize op."""
    base_name = input_node.name + "_"
    quint8_const_name = base_name + "quint8_const"
    min_name = base_name + "min"
    max_name = base_name + "max"
    float_tensor = tensor_util.MakeNdarray(input_node.attr["value"].tensor)
    min_value = np.min(float_tensor.flatten())
    max_value = np.max(float_tensor.flatten())
    # Make sure that the range includes zero.
    if min_value > 0.0:
        min_value = 0.0
    # min_value == max_value is a tricky case. It can occur for general
    # tensors, and of course for scalars. The quantized ops cannot deal
    # with this case, so we set max_value to something else.
    # It's a tricky question what is the numerically best solution to
    # deal with this degeneracy.
    # TODO(petewarden): Better use a tolerance than a hard comparison?
    if min_value == max_value:
        if abs(min_value) < 0.000001:
            max_value = min_value + 1.0
        elif min_value > 0:
            max_value = 2 * min_value
        else:
            max_value = min_value / 2.0

    sess = session.Session()
    with sess.as_default():
        quantize_op = array_ops.quantize_v2(
            float_tensor,
            min_value,
            max_value,
            dtypes.quint8,
            mode=quantization_mode)
        quint8_tensor = quantize_op[0].eval()
        min_value = quantize_op[1].eval()
        max_value = quantize_op[2].eval()
    shape = tensor_util.TensorShapeProtoToList(input_node.attr["value"] .tensor.tensor_shape)
    quint8_const_node = create_constant_node(
        quint8_const_name, quint8_tensor, dtypes.quint8, shape=shape)
   

    dtype = dtypes.as_dtype(input_node.attr["dtype"].type)
    min_node = create_constant_node(min_name, min_value, dtypes.float32)
    max_node = create_constant_node(max_name, max_value, dtypes.float32)    
    dequantize_node = create_node("Dequantize", input_node.name,
                                  [quint8_const_name, min_name, max_name])
    set_attr_dtype(dequantize_node, "T", dtypes.quint8)
    set_attr_string(dequantize_node, "mode", quantization_mode)
    return [quint8_const_node, min_node, max_node, dequantize_node]

def intel_cpu_quantize_weight_eightbit(input_node, quantization_mode="SCALED"):
    """Returns replacement of constant weight node.

    This function creates (i) a quantized constant node, (ii) a float min node
    (iii) a float max node, and (iv) a dequantize node."""
    base_name = input_node.name + "_"
    qint8_const_name = base_name + "qint8_const"
    min_name = base_name + "min"
    max_name = base_name + "max"
    float_tensor = tensor_util.MakeNdarray(input_node.attr["value"].tensor)
    min_value = np.min(float_tensor.flatten())
    max_value = np.max(float_tensor.flatten())
    # Same processing of min-max as in quantize_weight_eightbit function.
    if min_value > 0.0:
        min_value = 0.0
    if min_value == max_value:
        if abs(min_value) < 0.000001:
            max_value = min_value + 1.0
        elif min_value > 0:
            max_value = 2 * min_value
        else:
            max_value = min_value / 2.0

    sess = session.Session()
    with sess.as_default():
        quantize_op = array_ops.quantize_v2(
            float_tensor,
            min_value,
            max_value,
            dtypes.qint8,
            mode=quantization_mode,
            round_mode="HALF_TO_EVEN")
        qint8_tensor = quantize_op[0].eval()
        # Updated min-max values should be passed to the next feeding node.
        min_value = quantize_op[1].eval()
        max_value = quantize_op[2].eval()
    shape = tensor_util.TensorShapeProtoToList(input_node.attr["value"].tensor.tensor_shape)
    qint8_const_node = create_constant_node(
        qint8_const_name, qint8_tensor,
        dtypes.qint8,
        shape=shape)
    min_node = create_constant_node(min_name, min_value, dtypes.float32)
    max_node = create_constant_node(max_name, max_value, dtypes.float32)
   
    dequantize_node = create_node("Dequantize", input_node.name,
                                  [qint8_const_name, min_name, max_name])
    set_attr_dtype(dequantize_node, "T", dtypes.quint8)
    set_attr_string(dequantize_node, "mode", b'SCALED')
    return [qint8_const_node, min_node, max_node, dequantize_node]

EightbitizeRecursionState = collections.namedtuple(
    "EightbitizeRecursionState",
    ["already_visited", "output_node_stack", "merged_with_fake_quant"])

class GraphRewriter(object):
    """Takes a float graph, and rewrites it in quantized form."""

    def __init__(self,input_graph):
        """Sets up the class to rewrite a float graph.

        Args:
          input_graph: A float graph to transform.
          mode: A string controlling how quantization is performed -
            round, quantize, eightbit, or weights.
          quantized_input_range: if set, assume the input is
            quantized and represents the range
            [quantized_input_range[0], quantized_input_range[1]]
          fallback_quantization_range: if set, then for nodes where the quantization
            range can't be inferred from the graph, use the range
            [fallback_quantization_range[0], fallback_quantization_range[1]) instead
            of using a RequantizationRange node in the graph.
          excluded_ops: list of operations to be excluded from quantization
          excluded_nodes: list of nodes to be excluded from quantization

        Raises:
          ValueError: Two nodes with the same name were found in the graph.
        """
        self.input_graph = input_graph
        self.nodes_map = self.create_nodes_map(input_graph)
        self.output_graph = None
        self.conv_count = 0
        self.final_node_renames = {}
        self.quantized_node_dict = {}
        # Data that is valid only during the recursive call to rewrite the graph.
        self.state = None

    def create_nodes_map(self, graph):
        """Builds a mapping of node names to their defs from the graph."""
        nodes_map = {}
        for node in graph.node:
            if node.name not in nodes_map.keys():
              nodes_map[node.name] = node
            else:
             print(node.name)
             raise ValueError("Duplicate node names detected.")
        return nodes_map

    def should_merge_with_fake_quant_node(self):
        """Should the current node merge with self.state.output_node_stack[-1]?"""
        if not self.state.output_node_stack:
            return False
        top = self.state.output_node_stack[-1]
        return top[1] == 0 and top[0].op in ["FakeQuantWithMinMaxVars"]

    def should_quantize_const(self, node):
        if not self.state.output_node_stack:
            return False
        top = self.state.output_node_stack[-1]
        if not top[2]:
            return False
        dtype = dtypes.as_dtype(node.attr["dtype"].type)
        # assert dtype == dtypes.float32, (
        #  "Failed to quantized constant %s of type %s" % (node.name, dtype))
        return True

    # TODO(intel-tf): Quantized Matmul could be fused with few other succeeding
    # ops. Current support is for BiasAdd and Relu.
    def intel_cpu_eightbitize_matmul_node(self, original_node, bias_node=None,bias_add_name=None):
        """Replaces a matmul node with the eight bit equivalent sub-graph."""
        # print("******zjq*************inteleight**&&&&&&&&&&&&&&")
        all_input_names = self.add_eightbit_prologue_nodes_matmul(original_node)
        # quantize_bias = False
        if bias_node and bias_add_name:
            new_node = node_def_pb2.NodeDef()
            new_node.CopyFrom(bias_node)
            self.add_output_graph_node(new_node)
            all_input_names = all_input_names[:2] + [bias_node.name] + \
                    all_input_names[2:]
            quantized_mat_mul_name = original_node.name + "_eightbit_quantized_mat_mul"
            quantized_mat_mul_node = create_node("QuantizedMatMulWithBias", quantized_mat_mul_name, all_input_names)
            set_attr_dtype(quantized_mat_mul_node, "Tbias", dtypes.float32)
        set_attr_dtype(quantized_mat_mul_node, "T1", dtypes.quint8)
        set_attr_dtype(quantized_mat_mul_node, "T2", dtypes.qint8)
        set_attr_dtype(quantized_mat_mul_node, "Toutput", dtypes.qint32)
        copy_attr(quantized_mat_mul_node, "transpose_a",
                  original_node.attr["transpose_a"])
        copy_attr(quantized_mat_mul_node, "transpose_b",
                  original_node.attr["transpose_b"])
        self.add_output_graph_node(quantized_mat_mul_node)
        quantize_down_name = self.add_quantize_down_nodes(original_node, quantized_mat_mul_name)
        # print(bias_node)
        if bias_node and bias_add_name :
            # print(quantize_down_name+"\t"+bias_add_name)  #MatMul_eightbit_requantize      result
            self.add_dequantize_result_node(quantize_down_name, bias_add_name)
        else:
            self.add_dequantize_result_node(quantize_down_name, original_node.name)

    # TODO(intel-tf): We leave the output graph partially quantized for
    # intel cpu. Current quantization support is for Conv2D and its fusion.
    # More quantized operations will be included as more implementations are
    # completed.
    def intel_cpu_eightbitize_nodes_recursively(self, current_node):
        """The entry point for transforming a graph into full eight bit."""
        # print(current_node.name)
        if current_node.name in self.state.already_visited:
            if (self.should_merge_with_fake_quant_node() or
                    current_node.name in self.state.merged_with_fake_quant):
                raise ValueError("Unsupported graph structure: output of node %s "
                                 "is processed by a FakeQuant* node and should have "
                                 "no other outputs.", current_node.name)
            return
        self.state.already_visited[current_node.name] = True
 
        quantize_input, should_quantize_conv, \
            fuse_with_conv = (False, False, False)
        if current_node.op == "MatMul":
            should_quantize_conv = True
        inputs = list(enumerate(current_node.input))
        for i, input_node_name in inputs:
            input_node_name = node_name_from_input(input_node_name)
            input_node = self.nodes_map[input_node_name]
            if should_quantize_conv and i == 1 and input_node.op == "Const":
                quantize_input = True
            if current_node.op in ("MatMul") :
                quantize_input = True
            self.state.output_node_stack.append([current_node, i, quantize_input,  fuse_with_conv])
            #print(i)
            self.intel_cpu_eightbitize_nodes_recursively(input_node)
            self.state.output_node_stack.pop()

        
        #if ("lm/bert/encoder/layer_" in current_node.name and "/attention/self" in current_node.name)  or ("lm/bert/encoder/layer_" in current_node.name and "intermediate" in current_node.name) or "lm/bert/pooler" in current_node.name:
            #print(current_node.name)
        if "bert" in current_node.name:
            if current_node.op == "MatMul" and should_quantize_conv and quantize_input:
                # match pattern for fusion with bias and relu for Wide&Deep
                grand_parent, parent = self.state.output_node_stack[-2:]
                if parent[0].op == "BiasAdd" and (not self.state.output_node_stack[-2][3]) and (not "Switch" in str(current_node.input)):
                    self.state.output_node_stack[-2][3] = True  # BiasAdd to be fused
                    bias_node_name = node_name_from_input(parent[0].input[1])
                    bias_node = self.nodes_map[bias_node_name]
                    self.intel_cpu_eightbitize_matmul_node(current_node, bias_node, parent[0].name)
                else:    
                    new_node = node_def_pb2.NodeDef()
                    new_node.CopyFrom(current_node)
                    self.add_output_graph_node(new_node)
            
                
            elif current_node.op == "BiasAdd" and  self.state.output_node_stack[-1][3] == True :
                pass  # This op is already processed by fused quantization
            elif current_node.op == "Const" :
                parent = self.state.output_node_stack[-1]
                if parent[0].op == "MatMul" and parent[2]:
                    grand_parent = self.state.output_node_stack[-2]
                    if(grand_parent[0].op == "BiasAdd"):  
                     for n in intel_cpu_quantize_weight_eightbit(current_node, b"SCALED"):
                        self.add_output_graph_node(n)
                    else:
                        new_node = node_def_pb2.NodeDef()
                        new_node.CopyFrom(current_node)
                        self.add_output_graph_node(new_node)
                elif parent[0].op == "BiasAdd" and \
                        self.state.output_node_stack[-2][3]:
                    pass  # This constant is already process by fused quantization
                else:
                    new_node = node_def_pb2.NodeDef()
                    new_node.CopyFrom(current_node)
                    self.add_output_graph_node(new_node)
            else:
                new_node = node_def_pb2.NodeDef()
                new_node.CopyFrom(current_node)
                self.add_output_graph_node(new_node)
        else:
                new_node = node_def_pb2.NodeDef()
                new_node.CopyFrom(current_node)
                self.add_output_graph_node(new_node)

    def add_eightbit_prologue_nodes(self, original_node):
        """Adds input conversion nodes to handle quantizing the underlying node."""
        namespace_prefix = original_node.name + "_eightbit"

        # Use the name of the first input as the control input name
        # for reshape_dim and reduction_dim to slove the different frame issue
        # in quantized graph
        reshape_dims_name, reduction_dims_name = self.add_common_quantization_nodes(
            namespace_prefix, node_name_from_input(original_node.input[0]))
        input_names = []
        min_max_names = []
        for original_input_name in original_node.input:
            # Do not quantize control input
            if original_input_name[0] == '^':
                continue
            quantize_input_name, min_input_name, max_input_name = (
                self.eightbitize_input_to_node(namespace_prefix, original_input_name,
                                               reshape_dims_name,
                                               reduction_dims_name))
            input_names.append(quantize_input_name)
            min_max_names.append(min_input_name)
            min_max_names.append(max_input_name)
        all_input_names = []
        all_input_names.extend(input_names)
        all_input_names.extend(min_max_names)

        # add back control input name
        for original_input_name in original_node.input:
            if original_input_name[0] == '^':
                all_input_names.append(original_input_name)

        return all_input_names

    """Get input to matmul node  """

    def add_eightbit_prologue_nodes_matmul(self, original_node):
        """Adds input conversion nodes to handle quantizing the underlying node."""
        namespace_prefix = original_node.name + "_eightbit"
        reshape_dims_name, reduction_dims_name = self.add_common_quantization_nodes(
            namespace_prefix)
        input_names = []
        min_max_names = []
        for original_input_name in original_node.input:
            quantize_input_name, min_input_name, max_input_name = (self.eightbitize_input_to_node_matmul(namespace_prefix, original_input_name,
                                                      reshape_dims_name,
                                                      reduction_dims_name))
            input_names.append(quantize_input_name)
            min_max_names.append(min_input_name)
            min_max_names.append(max_input_name)
        all_input_names = []
        all_input_names.extend(input_names)
        all_input_names.extend(min_max_names)
        return all_input_names

    def add_common_quantization_nodes(self, namespace_prefix, control_input_name=None):
        """Builds constant nodes needed for quantization of inputs."""

        reshape_dims_name = namespace_prefix + "_reshape_dims"
        reduction_dims_name = namespace_prefix + "_reduction_dims"

        reshape_dims_node = create_constant_node(reshape_dims_name, -1,dtypes.int32, [1])
        if control_input_name:
            reshape_dims_node.input.append("^" + control_input_name)
        self.add_output_graph_node(reshape_dims_node)
        reduction_dims_node = create_constant_node(reduction_dims_name, 0,
                                                   dtypes.int32, [1])
        if control_input_name:
            reduction_dims_node.input.append("^" + control_input_name)
        self.add_output_graph_node(reduction_dims_node)
        return reshape_dims_name, reduction_dims_name

    def eightbitize_input_to_node(self, namespace_prefix, original_input_name,
                                  reshape_dims_name, reduction_dims_name):
        """Takes one float input to an op, and converts it to quantized form."""
        unique_input_name = unique_node_name_from_input(original_input_name)
        if unique_input_name in self.quantized_node_dict:
            quantized_tuple = self.quantized_node_dict[unique_input_name]
            return quantized_tuple[0], quantized_tuple[1], quantized_tuple[2]

        reshape_input_name = namespace_prefix + "_reshape_" + unique_input_name
        min_input_name = namespace_prefix + "_min_" + unique_input_name
        max_input_name = namespace_prefix + "_max_" + unique_input_name
        quantize_input_name = namespace_prefix + "_quantize_" + unique_input_name
        reshape_input_node = create_node("Reshape", reshape_input_name,
                                         [original_input_name, reshape_dims_name])
        set_attr_dtype(reshape_input_node, "T", dtypes.float32)
        self.add_output_graph_node(reshape_input_node)
        min_input_node = create_node("Min", min_input_name,
         [reshape_input_name, reduction_dims_name])
        set_attr_dtype(min_input_node, "T", dtypes.float32)
        set_attr_dtype(min_input_node, "Tidx", dtypes.int32)
        set_attr_bool(min_input_node, "keep_dims", False)
        self.add_output_graph_node(min_input_node)
        max_input_node = create_node("Max", max_input_name,
                                     [reshape_input_name, reduction_dims_name])
        set_attr_dtype(max_input_node, "T", dtypes.float32)
        set_attr_dtype(max_input_node, "Tidx", dtypes.int32)
        set_attr_bool(max_input_node, "keep_dims", False)
        self.add_output_graph_node(max_input_node)
        quantize_input_node = create_node(
            "QuantizeV2", quantize_input_name,
            [original_input_name, min_input_name, max_input_name])
        set_attr_dtype(quantize_input_node, "T", dtypes.quint8)
        set_attr_string(quantize_input_node, "mode",
                        b"SCALED" if self.intel_cpu_eightbitize else b"MIN_FIRST")
        set_attr_string(quantize_input_node, "round_mode",
                        b"HALF_TO_EVEN" if self.intel_cpu_eightbitize
                        else b"HALF_AWAY_FROM_ZERO")
        self.add_output_graph_node(quantize_input_node)
        min_output_name = quantize_input_name + ":1"
        max_output_name = quantize_input_name + ":2"

        self.quantized_node_dict[unique_input_name] = (quantize_input_name,
                                                       min_output_name, max_output_name)
        return quantize_input_name, min_output_name, max_output_name

    """Adding this function for Wide and Deep to quantize inputs in MIN_FIRST mode"""

    def eightbitize_input_to_node_matmul(self, namespace_prefix, original_input_name,
                                         reshape_dims_name, reduction_dims_name):
        """Takes one float input to an op, and converts it to quantized form."""
        unique_input_name = unique_node_name_from_input(original_input_name)
        # print(namespace_prefix+",**********namespace_prefix") #a_name_scope/MatMul_eightbit
        # print(unique_input_name+",**********unique_input_name")  #a_name_scope/x
        # print(original_input_name+",******original_input_name") #a_name_scope/x,******original_input_name
        # unique_input_name = ""
        reshape_input_name = namespace_prefix + "_reshape_" + unique_input_name
        min_input_name = namespace_prefix + "_min_" + unique_input_name
        max_input_name = namespace_prefix + "_max_" + unique_input_name
        quantize_input_name = namespace_prefix + "_quantize_" + unique_input_name
        reshape_input_node = create_node("Reshape", reshape_input_name,
                                         [original_input_name, reshape_dims_name])
        # print(reshape_input_node.name+",*****reshape_input_node.name") #MatMul_eightbit_reshape_x
        set_attr_dtype(reshape_input_node, "T", dtypes.float32)
        self.add_output_graph_node(reshape_input_node)

        min_input_node = create_node("Min", min_input_name,
                                     [reshape_input_name, reduction_dims_name])
        set_attr_dtype(min_input_node, "T", dtypes.float32)
        set_attr_dtype(min_input_node, "Tidx", dtypes.int32)
        set_attr_bool(min_input_node, "keep_dims", False)
        self.add_output_graph_node(min_input_node)
        max_input_node = create_node("Max", max_input_name,
                                     [reshape_input_name, reduction_dims_name])
        set_attr_dtype(max_input_node, "T", dtypes.float32)
        set_attr_dtype(max_input_node, "Tidx", dtypes.int32)
        set_attr_bool(max_input_node, "keep_dims", False)
        self.add_output_graph_node(max_input_node)
        quantize_input_node = create_node(
            "QuantizeV2", quantize_input_name,
            [original_input_name, min_input_name, max_input_name])
        set_attr_dtype(quantize_input_node, "T", dtypes.quint8)
        set_attr_string(quantize_input_node, "mode", b"MIN_FIRST")
        self.add_output_graph_node(quantize_input_node)
        min_output_name = quantize_input_name + ":1"
        max_output_name = quantize_input_name + ":2"
        return quantize_input_name, min_output_name, max_output_name

    def add_quantize_down_nodes(self, original_node, quantized_output_name):
         quantized_outputs = [
            quantized_output_name, quantized_output_name + ":1",
            quantized_output_name + ":2"
         ]
         min_max_inputs = None
         if self.should_merge_with_fake_quant_node():
            # Use the inputs to the FakeQuantWithMinMaxVars node as the inputs to
            # Requantize.
            fake_quant_node = self.state.output_node_stack[-1][0]
            min_max_inputs = [fake_quant_node.input[1], fake_quant_node.input[2]]
            assert original_node.name not in self.state.merged_with_fake_quant
            self.state.merged_with_fake_quant[original_node.name] = True
         else:
            # Add a RequantizationRange node for finding the min and max values.
            requant_range_node = create_node(
                "RequantizationRange", original_node.name + "_eightbit_requant_range",
                quantized_outputs)
            set_attr_dtype(requant_range_node, "Tinput", dtypes.qint32)
            self.add_output_graph_node(requant_range_node)
            min_max_inputs = [
                requant_range_node.name + ":0", requant_range_node.name + ":1"
            ]
         requantize_node = create_node("Requantize",
                                      original_node.name + "_eightbit_requantize",
                                      quantized_outputs + min_max_inputs)
         set_attr_dtype(requantize_node, "Tinput", dtypes.qint32)
         set_attr_dtype(requantize_node, "out_type", dtypes.quint8)
         self.add_output_graph_node(requantize_node)
         return requantize_node.name

    def add_dequantize_result_node(self,quantized_output_name,original_node_name,min_tensor_index=1):
        min_max_inputs = [
            "%s:%s" % (quantized_output_name, min_tensor_index),
            "%s:%s" % (quantized_output_name, (min_tensor_index + 1))
        ]
        dequantize_name = original_node_name
        if self.should_merge_with_fake_quant_node():
            fake_quant_node = self.state.output_node_stack[-1][0]
            if original_node_name not in self.state.merged_with_fake_quant:
                min_max_inputs = [fake_quant_node.input[1], fake_quant_node.input[2]]
                self.state.merged_with_fake_quant[original_node_name] = True
            dequantize_name = fake_quant_node.name

        dequantize_node = create_node(
            "Dequantize", dequantize_name,
            [quantized_output_name, min_max_inputs[0], min_max_inputs[1]])
        set_attr_dtype(dequantize_node, "T", dtypes.quint8)
        set_attr_string(dequantize_node, "mode", b"MIN_FIRST")
        # print(dequantize_node.name+",****")
        self.add_output_graph_node(dequantize_node)

    def add_output_graph_node(self, output_node):
        """Inserts one node into the new graph."""
        self.output_graph.node.extend([output_node])

    def remove_redundant_quantization(self, old_graph):
        """Removes unneeded pairs of quantize/dequantize ops from the graph.

        This is a bit of a tricky function, because it's attempting to spot the
        pattern of dequantizing from eight-bit up to float, and then immediately
        quantizing back down to eight bits again, that's introduced by previous
        passes that do 'key-hole' conversions of individual nodes but have to
        convert back to float to match the previous output interface, since they
        don't know that the next op can handle quantized tensors.
        It works by:
         - Looking for Quantize nodes.
         - Checking to see if their first input is a Dequantize node.
         - Seeing if their min/max inputs come from Min/Max nodes.
         - Making sure those Min/Max nodes are being fed from the same Dequantize.
         - Or that the Min is indirectly being fed from the same Dequantize as Max.
         - Making sure the Dequantize is going through a Reshape (which we add
           during the previous pass when we create the quantize sub-graph).
         - Looking for the dims Const op for the Min/Max dims.
        If all of these conditions are met, then it's a sub-graph pattern that
        we know how to optimize out (and is likely the common one we've introduced).
        We then rewire the graph to skip it entirely, and then rely on the dead node
        removal pass to get rid of any nodes that are no longer needed.

        Args:
          old_graph: The model we'll be stripping redundant nodes from.

        Returns:
          A graph with the unnecessary nodes removed.

        Raises:
          ValueError: Two nodes with the same name were found in the graph.
        """
        old_nodes_map = self.create_nodes_map(old_graph)
        self.output_graph = graph_pb2.GraphDef()
        inputs_to_rename = {}
        # We go through all the nodes, looking for any that match the patterns we
        # know how to optimize away.
        for node in old_graph.node:
            # We always start with a Quantize node, and examine its inputs to see if
            # they are in a form that can be removed.
            if node.op not in ["Quantize", "QuantizeV2"]:
                continue
            dequantize_node_name = node_name_from_input(node.input[0])
            if dequantize_node_name not in old_nodes_map:
                raise ValueError("Input node name '" + dequantize_node_name +
                                 "' not found in node '" + node.name + "'")
            dequantize_node = old_nodes_map[dequantize_node_name]
            # Do we have a Dequantize feeding in, with the same type as the Quantize?
            if dequantize_node.op != "Dequantize":
                continue
            if node.attr["T"] != dequantize_node.attr["T"]:
                continue
            # Now look at the other inputs, and ensure they're Min/Max nodes.
            min_node_name = node_name_from_input(node.input[1])
            max_node_name = node_name_from_input(node.input[2])
            min_node = old_nodes_map[min_node_name]
            max_node = old_nodes_map[max_node_name]
            is_min_right_type = (min_node.op in ["Min", "Dequantize"])
            is_max_right_type = (max_node.op in ["Max", "Dequantize"])
            if not is_min_right_type or not is_max_right_type:
                print("Didn't find expected types on inputs : %s, %s." % (min_node.op,max_node.op))
                continue
            min_node_input_name = node_name_from_input(min_node.input[0])
            max_node_input_name = node_name_from_input(max_node.input[0])
             # There are two different patterns for Min nodes we can recognize, one
            # where the input comes directly from the same one as the Max, and
            # another where we run it through another Min first, so check for both.
            is_same_input = False
            if min_node_input_name == max_node_input_name:
                is_same_input = True
            else:
                first_min_node_input = old_nodes_map[min_node_input_name]
                if first_min_node_input.op == "Concat":
                    second_min_node_name = node_name_from_input(
                        first_min_node_input.input[1])
                    second_min_node = old_nodes_map[second_min_node_name]
                    if second_min_node.op == "Min":
                        second_min_node_input_name = node_name_from_input(
                            second_min_node.input[0])
                        is_same_input = (second_min_node_input_name == max_node_input_name)
            if not is_same_input:
                print("Different min/max inputs: " + min_node_input_name)
                continue
            # We recognize this pattern, so mark the graph edges to be rewired to
            # route around it entirely, since we know it's a no-op.
            dequantize_source_name = node_name_from_input(dequantize_node.input[0])
            node_tensor_name = ensure_tensor_name_has_port(node.name)
            min_tensor_name = node.name + ":1"
            max_tensor_name = node.name + ":2"
            inputs_to_rename[node_tensor_name] = dequantize_source_name
            inputs_to_rename[min_tensor_name] = dequantize_node.input[1]
            inputs_to_rename[max_tensor_name] = dequantize_node.input[2]
        # Finally we apply all the rewiring we've marked to the graph.
        for node in old_graph.node:
            for index, input_full_name in enumerate(node.input):
                input_name = ensure_tensor_name_has_port(input_full_name)
                if input_name in inputs_to_rename:
                    node.input[index] = inputs_to_rename[input_name]
            self.add_output_graph_node(node)
        return self.output_graph

    def apply_final_node_renames(self):
        """Applies node renames in self.final_node_renames to self.output_graph."""
        old_graph = self.output_graph
        self.output_graph = graph_pb2.GraphDef()
        for node in old_graph.node:
            node.name = self.final_node_renames.get(node.name, node.name)
            for index, input_name in enumerate(node.input):
                node_name = node_name_from_input(input_name)
                input_full_name = ensure_tensor_name_has_port(input_name)
                if node_name in self.final_node_renames:
                    node.input[index] = "%s%s" % (self.final_node_renames[node_name],
                                                  input_full_name[len(node_name):])
            self.add_output_graph_node(node)
        return self.output_graph

    def remove_dead_nodes(self, output_names):
        """Removes nodes that are no longer needed for inference from the graph."""
        old_output_graph = self.output_graph
        self.output_graph = graph_util.extract_sub_graph(old_output_graph,output_names)

    def set_input_graph(self, new_input_graph):
        self.input_graph = new_input_graph
        self.nodes_map = self.create_nodes_map(self.input_graph)

    def rewrite(self, output_node_names):
        """Triggers rewriting of the float graph.

        Args:
          output_node_names: A list of names of the nodes that produce the final
            results.

        Returns:
          A quantized version of the float graph.
        """
        self.output_graph = graph_pb2.GraphDef()
        output_nodes = [
            self.nodes_map[output_node_name]
            for output_node_name in output_node_names
        ]
        # output_nodes = []
        # for output_node_name in output_node_names:
        #   output_nodes.append(self.nodes_map[output_node_name])

        # When function graph_util.remove_training_nodes remove
        # "Identity" ops in the graph, it does not replace the
        # control input properly, so the control input becomes
         # the regular input. Disable this function until the
        # the bug is fixed.

        self.set_input_graph(graph_util.remove_training_nodes(
            self.input_graph, protected_nodes=output_node_names))

        output_nodes = [
            self.nodes_map[output_node_name]
            for output_node_name in output_node_names
        ]

        # output_nodes=[]
        # for output_node_name in output_node_names:
        #   output_nodes.append(self.nodes_map[output_node_name])

        self.state = EightbitizeRecursionState(
            already_visited={}, output_node_stack=[], merged_with_fake_quant={})

            # TODO(intel-tf): Enables fused quantized node for intel cpu.
        for output_node in output_nodes:
            # Intiailize output_node_stack with output node.
            # Each element in the stack is a mutable list containing
            # [parent_node, index_to_parent, quantization_flag, fusion_flag].
            # In case of root node, make self as parent.
            self.state.output_node_stack.append(
                [output_node, None, False, False])
            self.intel_cpu_eightbitize_nodes_recursively(output_node)
            self.state.output_node_stack.pop()

        self.state = None
        if strip_redundant_quantization:
            self.output_graph = self.remove_redundant_quantization(
                self.output_graph)
            self.remove_dead_nodes(output_node_names)
        self.apply_final_node_renames()
        return self.output_graph



def main(unused_args):
    tf_graph = graph_pb2.GraphDef()
#  with gfile.Open("matmul_model.pb", "rb") as f:
#     tf_graph.ParseFromString(f.read())
    with gfile.Open("model_new.pb", "rb") as f:
        data = f.read()
        tf_graph.ParseFromString(data)

    graph = ops.Graph()
    with graph.as_default():
        importer.import_graph_def(tf_graph,input_map={},name="")
    rewriter = GraphRewriter(tf_graph)
    # f = gfile.FastGFile("model_slot_intel.pb", "wb")
    # output_graph = rewriter.rewrite(['input_idx','length','keep_prob','is_training','input_mask','slot_tagging/tagging_output','slot_tagging/crf_layer/transitions'])
    # f = gfile.FastGFile("matmul_model_intel.pb", "wb")
    # output_graph = rewriter.rewrite(['result'])
    f = gfile.FastGFile("model_intel_all.pb", "wb")
    output_graph = rewriter.rewrite(['score','predict_res'])
    f.write(output_graph.SerializeToString())

if __name__ == "__main__":
    app.run()
                             
