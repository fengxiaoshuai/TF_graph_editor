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

def create_constant_node(name, value, dtype, shape=None):
    node = create_node("Const", name, [])
    set_attr_dtype(node, "dtype", dtype)
    set_attr_tensor(node, "value", value, dtype, shape)
    return node

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


NodeState = collections.namedtuple( "NodeState",["already_visited","output_node_stack"])

class GraphRewriter(object):
    """Takes a float graph, and rewrites it in quantized form."""

    def __init__(self,input_graph):
        """Sets up the class to rewrite a float graph.

        """
        self.input_graph = input_graph
        self.nodes_map = self.create_nodes_map(input_graph)
        self.state = None
        self.final_node_renames = {}
        self.output_graph = None
        self.exclude_list = ['None']

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


    def replace_nodes_recursively(self, current_node):
        if current_node.name in self.state.already_visited:
          return
        self.state.already_visited[current_node.name] = True         

        inputs = list(enumerate(current_node.input))
        for i, input_node_name in inputs:
            input_node_name = node_name_from_input(input_node_name)
            input_node = self.nodes_map[input_node_name]
            self.state.output_node_stack.append([current_node, i])
            self.replace_nodes_recursively(input_node)
            self.state.output_node_stack.pop()
        if "bert" in current_node.name:
            if current_node.op == "Pow":
                gelu_op_list = self.state.output_node_stack[-7:]
                gelu_parent = self.state.output_node_stack[-8]
                for i in range(len(gelu_op_list)):
                  self.exclude_list.append(gelu_op_list[i][0].name)
                self.exclude_list.append(gelu_parent[0].name)
                self.replace_gelu_op(current_node,gelu_parent[0])
            elif current_node.op == "Softmax":
                softmax_parent = self.state.output_node_stack[-1]
                if "MatMul_1" in softmax_parent[0].name:
                  self.exclude_list.append(softmax_parent[0].name)
                  self.replace_softmax_dfour_op(current_node,softmax_parent[0])
            elif "output/add" in current_node.name:              #op操作是add
                add_parent = self.state.output_node_stack[-1]
                if "LayerNorm/batchnorm" in add_parent[0].name:  #并且父节点的name是LayerNorm中
                  self.exclude_list.append(current_node.name)    #当前add节点注释
                  self.replace_postlayernorm_op(current_node)    #替换layer_norm节点
            elif "attention/self/key/MatMul" in current_node.name or "attention/self/value/MatMul" in current_node.name:
                  self.replace_postlayernorm_parent_op(current_node)
            elif "LayerNorm/batchnorm/add_1"  in current_node.name:  #
                 layernorm_parent = self.state.output_node_stack[-1]
                 if "embeddings" in current_node.name:
                    self.replace_prelayernorm_parent_op(layernorm_parent[0])
                 else:
                    self.replace_postlayernorm_parent_op(layernorm_parent[0])
            elif "LayerNorm/moments" in current_node.name or "LayerNorm/batchnorm" in current_node.name:  #将这两个子图给去掉
                 self.exclude_list.append(current_node.name)
            elif "embeddings/add_1" in current_node.name:
                 self.replace_prelayernorm_op(current_node)
            elif "bert/encoder/Shape_2" in current_node.name:
                 self.replace_layernorm_parent_shape_op(current_node)
            elif current_node.name in self.exclude_list:
                 pass
            else:
                new_node = node_def_pb2.NodeDef()
                new_node.CopyFrom(current_node)
                self.add_output_graph_node(new_node)
        else:
                new_node = node_def_pb2.NodeDef()
                new_node.CopyFrom(current_node)
                self.add_output_graph_node(new_node)

    def replace_gelu_op(self,current_node,gelu_parent):
       gelu_node = node_def_pb2.NodeDef()
       prefix = current_node.name.rsplit('/',1)[0]
       gelu_node.op = "GeluErfActivation"
       gelu_node.name = prefix + "/" +"GeluErfActivation"
       gelu_node.input.extend([current_node.input[0]])
       self.add_output_graph_node(gelu_node)
        
       gelu_parent_new = node_def_pb2.NodeDef()
       gelu_parent_new.op = gelu_parent.op
       gelu_parent_new.name = gelu_parent.name
       gelu_parent_new.attr["transpose_a"].CopyFrom(gelu_parent.attr["transpose_a"])
       gelu_parent_new.attr["transpose_b"].CopyFrom(gelu_parent.attr["transpose_b"])
       gelu_parent_new.attr["T"].CopyFrom(gelu_parent.attr["T"])
       input_1 = gelu_parent.input[1]
       gelu_parent_new.input.extend([gelu_node.name,input_1])
       self.add_output_graph_node(gelu_parent_new) 
      
 
    def replace_softmax_dfour_op(self,current_node,softmax_parent):
       softmax_dfour_node = node_def_pb2.NodeDef()
       prefix = current_node.name.rsplit('/',1)[0]
       softmax_dfour_node.op = "OptSoftmaxDfour"
       softmax_dfour_node.name = prefix + "/" +"OptSoftmaxDfour"
       softmax_dfour_node.input.extend([current_node.input[0]])
       self.add_output_graph_node(softmax_dfour_node)

       softmax_parent_new = node_def_pb2.NodeDef()
       softmax_parent_new.op = softmax_parent.op
       softmax_parent_new.name = softmax_parent.name
       softmax_parent_new.attr["adj_x"].CopyFrom(softmax_parent.attr["adj_x"])
       softmax_parent_new.attr["adj_y"].CopyFrom(softmax_parent.attr["adj_y"])
       softmax_parent_new.attr["T"].CopyFrom(softmax_parent.attr["T"])
       input_1 = softmax_parent.input[1]
       softmax_parent_new.input.extend([softmax_dfour_node.name,input_1])
       self.add_output_graph_node(softmax_parent_new)
    
    def replace_postlayernorm_op(self,current_node):
      input_0 = current_node.input[0]
      input_3 = current_node.input[1]
      if "LayerNorm/batchnorm" in input_3:
         input_3 = current_node.input[1].rsplit('/',3)[0]+"/"+"OptLayerPostprocessDtwo"
      postprocess_dtwo_node = node_def_pb2.NodeDef()
      prefix = current_node.name.rsplit('/',1)[0]
      postprocess_dtwo_node.name = prefix + "/" + "OptLayerPostprocessDtwo"
      postprocess_dtwo_node.op = "OptLayerPostprocessDtwo"
      input_1 = prefix + "/" + "LayerNorm/gamma/read"
      input_2 = prefix + "/" + "LayerNorm/beta/read"
      postprocess_dtwo_node.input.extend([input_0,input_1,input_2,input_3])
      self.add_output_graph_node(postprocess_dtwo_node)
    
    def replace_postlayernorm_parent_op(self,current_node):
        if "LayerNorm/batchnorm/add_1" in current_node.input[0]:
           input_1 = current_node.input[1]
           input_0 = current_node.input[0].rsplit('/',3)[0]+"/"+"OptLayerPostprocessDtwo"
           current_node.input[:] = [input_0,input_1] 
        if "attention/self/key" in current_node.name or "attention/self/value" in current_node.name:
           print(current_node)
          
           self.add_output_graph_node(current_node)

    def replace_prelayernorm_op(self,current_node):
      input_0 = current_node.name
      self.add_output_graph_node(current_node)
      preprocess_dtwo_node = node_def_pb2.NodeDef()
      prefix = current_node.name.rsplit('/',1)[0]
      preprocess_dtwo_node.name = prefix + "/" + "OptLayerPreprocessDthree"
      preprocess_dtwo_node.op = "OptLayerPreprocessDthree"
      input_1 = prefix + "/" + "LayerNorm/gamma/read"
      input_2 = prefix + "/" + "LayerNorm/beta/read"
      input_3 = prefix + "/" + "Const"
      const_node = create_constant_node(input_3,1e-6,dtypes.float32)
      self.add_output_graph_node(const_node)
      preprocess_dtwo_node.input.extend([input_0,input_1,input_2,input_3])
      self.add_output_graph_node(preprocess_dtwo_node)
 
    def replace_layernorm_parent_shape_op(self,current_node):
        input_0 = current_node.input[0].rsplit('/',3)[0]+"/"+"OptLayerPreprocessDthree"
        current_node.input[:] = [input_0]
        self.add_output_graph_node(current_node)
 
    def replace_prelayernorm_parent_op(self,layernorm_parent_node):
         if "LayerNorm/batchnorm/add_1" in layernorm_parent_node.input[0]:
           input_1 = layernorm_parent_node.input[1]
           input_0 = layernorm_parent_node.input[0].rsplit('/',3)[0]+"/"+"OptLayerPreprocessDthree"
           layernorm_parent_node.input[:] = [input_0,input_1]
         #self.add_output_graph_node(layernorm_parent_node)


    def add_output_graph_node(self, output_node):
        """Inserts one node into the new graph."""
        self.output_graph.node.extend([output_node])


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
        #print(output_nodes)
        self.state = NodeState(
            already_visited={}, output_node_stack=[])
        for output_node in output_nodes:
            # Intiailize output_node_stack with output node.
            # Each element in the stack is a mutable list containing
            # [parent_node, index_to_parent, quantization_flag, fusion_flag].
            # In case of root node, make self as parent.
            self.state.output_node_stack.append([output_node,None])
            self.replace_nodes_recursively(output_node)
            #print(str(self.output_node_stack.pop())+"********************")
            self.state.output_node_stack.pop()

        self.apply_final_node_renames()
        return self.output_graph



def main(unused_args):
    tf_graph = graph_pb2.GraphDef()
#  with gfile.Open("matmul_model.pb", "rb") as f:
#     tf_graph.ParseFromString(f.read())
    with gfile.Open("diyCommentTagmodel.pb", "rb") as f:
        data = f.read()
        tf_graph.ParseFromString(data)

    graph = ops.Graph()
    with graph.as_default():
        importer.import_graph_def(tf_graph,input_map={},name="")
    rewriter = GraphRewriter(tf_graph)
    f = gfile.FastGFile("diy_userop.pb", "wb")
    output_graph = rewriter.rewrite(['score','predict_res'])
    f.write(output_graph.SerializeToString())

if __name__ == "__main__":
    app.run()
                             
