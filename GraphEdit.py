import os
import numpy as np
import collections
import tensorflow as tf
from tensorflow.core.framework import tensor_shape_pb2


class GraphEdit(object):
    def __init__(self, graph_pb, input_node, output_node):
        """
        :param graph_pb: the dir of pb
        :param input_node: the list of input_name
        :param output_node: the list of output_name
        """
        self.graph_pb = self.read_pb(graph_pb)
        self.node_name = [x.name for x in self.graph_pb.node]
        self.input_node_name = self.read_node(input_node)
        self.output_node_name = self.read_node(output_node)
        self.node_reference_count = self.reference_count()
        self.new_graph = tf.GraphDef()

    def __repr__(self):
        """
        :return: print input_node and output_node
        """
        return 'input_node:' + str(self.input_node_name) + '\n' + \
               'output_node:' + str(self.output_node_name)

    def reference_count(self):
        """
        :return:  node_reference_count
        """
        node_reference_count = collections.defaultdict(int)
        for node in self.graph_pb.node:
            for input_name in node.input:
                if input_name.startswith("^"):
                    input_name = input_name[1:]
                node_reference_count[input_name] += 1
        for output_name in self.output_node_name:
            node_reference_count[output_name] += 1
        return node_reference_count

    @staticmethod
    def read_pb(pb_dir):
        """
        :param pb_dir: the dir of old_vgg_pb
        :return: graph_def
        """
        try:
            f = open(pb_dir)
            f.close()
        except IOError:
            print("File is not accessible.")
            pb_dir = input('Correct Src:')
        # read file
        graph_def = tf.GraphDef()
        with tf.gfile.FastGFile(pb_dir, 'rb') as f:
            graph_def.ParseFromString(f.read())
        return graph_def

    @staticmethod
    def read_node(node_list):
        assert isinstance(node_list, list), "The type of '{}' must be list.".format(node_list)
        return node_list

    def search_node(self, node_name):
        assert node_name in self.node_name, "There isn't '{}' in the graph.".format(node_name)
        for node in self.graph_pb.node:
            if node.name == node_name:
                return node

    def replace_node(self, old_node_name, new_node):
        """
        the following operations are supported
        ----------------------------------------------
         head_op------>op_1------>tail_op
        ----------------------------------------------
        head_op------->op_2------>tail_op
        ----------------------------------------------
        :param old_node_name: the name of old_node
        :param new_node: must be type of tf.NodeDef
        :return: a new graph_def
        """

        if isinstance(new_node, tf.NodeDef):
            # Initialize old_node
            old_node = tf.NodeDef()
            for node in self.graph_pb.node:
                if node.name == old_node_name:
                    old_node = node
                    break
            if old_node.name != old_node_name:
                print("there isn't this node in graph")
                return

            # change the input of new_node
            for input_name in old_node.input:
                new_node.input.extend([input_name])

            # route the new_node
            for item in self.graph_pb.node:
                for i, _name in enumerate(item.input):
                    if old_node.name == _name:
                        item.input[i] = new_node.name

            # remove old_node
            self.node_reference_count[old_node.name] = 0

            # build new_graph
            for node in self.graph_pb.node:
                if self.node_reference_count[node.name] < 1:
                    continue
                new = tf.NodeDef()
                new.CopyFrom(node)
                self.new_graph.node.extend([new])
            self.new_graph.node.extend([new_node])
            return self.new_graph
        else:
            print("new_node must be the type of tf.NodeDef")
            return

    def init_node(self, node_name):
        """
        according to the node_name, find the node from the graph
        :param node_name:
        :return:
        """
        new_node = tf.NodeDef()
        for node in self.graph_pb.node:
            if node.name == node_name:
                new_node = node
                return new_node
        if new_node.name != node_name:
            print("There isn't this node in graph")
            return

    def add_node(self, head_node_name, tail_node_name, new_node):
        """
        the following operations are supported
        ----------------------------------------------
         head_op------->tail_op
        ----------------------------------------------
        head_op----->add_op------>tail_op
        ----------------------------------------------
        :param head_node_name:
        :param tail_node_name:
        :param new_node:
        :return:
        """

        if isinstance(new_node, tf.NodeDef):
            # Initialize head_node_name and tail_node_name

            head_node = self.init_node(head_node_name)
            tail_node = self.init_node(tail_node_name)

            # extend the input
            new_node.input.extend([head_node.name])

            # route the new_node
            for item in self.graph_pb.node:
                if item.name == tail_node.name:
                    for i, _name in enumerate(item.input):
                        if head_node_name == _name:
                            item.input[i] = new_node.name

            # build new_graph
            for node in self.graph_pb.node:
                if self.node_reference_count[node.name] < 1:
                    continue
                new = tf.NodeDef()
                new.CopyFrom(node)
                self.new_graph.node.extend([new])
            self.new_graph.node.extend([new_node])
            return self.new_graph
        else:
            print("New_node must be the type of tf.NodeDef")
            return

    def delete_node(self, node_name):
        """
        the following operations are supported
        ----------------------------------------------
                         |------>op_1
         op----->remove_op------>op_2
                        |------>op_3
        ----------------------------------------------
          |------>op_1
         op----->op_2
        |------>op_3
        ----------------------------------------------
        :param node_name:
        :return:
        """

        # init remove_node
        remove_node = self.init_node(node_name)

        # remove old_node
        assert node_name in self.node_name, "This node isn't in graph"
        self.node_reference_count[node_name] = 0

        # route the new_node
        for item in self.graph_pb.node:
            for i, _name in enumerate(item.input):
                if remove_node.name == _name:
                    item.input[i] = remove_node.input[0]

        # build new_graph
        for node in self.graph_pb.node:
            if self.node_reference_count[node.name] < 1:
                continue
            new = tf.NodeDef()
            new.CopyFrom(node)
            self.new_graph.node.extend([new])
        return self.new_graph

    def save(self, output_node, dst):
        try:
            if os.path.splitext(dst)[-1] != '.pb':
                raise RuntimeError('testError')
        except RuntimeError:
            print("Please input right dst, for example:'./hello.pb'")
            dst = input('Right Dst:')
        # build a new graph
        new_graph = tf.Graph()
        with new_graph.as_default():
            tf.import_graph_def(self.new_graph, name='')  # Imports `graph_def` into the current default `Graph`
        with tf.Session(graph=new_graph) as sess:
            frozen = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, output_node)
            with tf.gfile.FastGFile(dst, mode='wb') as f:
                f.write(frozen.SerializeToString())


class BuildNode(object):
    """
    The function of this class is to provide some graph nodes
    for example:
        new_node = BuildNode().creat_conv_node('test', stride=[1,2,2,1], padding=b'SAME')
    """

    @staticmethod
    def creat_conv_node(op_name, stride, padding=b'VALID', dtype=tf.float32):
        """
        :param op_name:
        :param stride:
        :param padding:
        :param dtype:
        :return:
        """

        new_node = tf.NodeDef()
        new_node.op = 'Conv2D'
        new_node.name = op_name
        new_node.attr["T"].CopyFrom(tf.AttrValue(type=dtype.as_datatype_enum))
        new_node.attr["use_cudnn_on_gpu"].CopyFrom(tf.AttrValue(b=1))
        new_node.attr["strides"].CopyFrom(tf.AttrValue(list=tf.AttrValue.ListValue(i=stride)))
        new_node.attr["padding"].CopyFrom(tf.AttrValue(s=padding))
        return new_node

    @staticmethod
    def creat_maxpool_node(op_name, ksize, stride, padding=b'VALID', dtype=tf.float32):
        """
        :param op_name:
        :param ksize:
        :param stride:
        :param padding:
        :param dtype:
        :return:
        """
        new_node = tf.NodeDef()
        new_node.op = 'MaxPool'
        new_node.name = op_name
        new_node.attr["ksize"].CopyFrom(tf.AttrValue(list=tf.AttrValue.ListValue(i=ksize)))
        new_node.attr["T"].CopyFrom(tf.AttrValue(type=dtype.as_datatype_enum))
        new_node.attr["strides"].CopyFrom(tf.AttrValue(list=tf.AttrValue.ListValue(i=stride)))
        new_node.attr["padding"].CopyFrom(tf.AttrValue(s=padding))
        return new_node

    @staticmethod
    def creat_mul_node(op_name, dtype=tf.float32):
        """
        :param op_name:
        :param dtype: tf.float32,tf.int8
        :return:
        """
        new_node = tf.NodeDef()
        new_node.op = 'Mul'
        new_node.name = op_name
        new_node.attr["T"].CopyFrom(tf.AttrValue(type=dtype.as_datatype_enum))
        return new_node

    @staticmethod
    def creat_const_node(op_name, arrary=None, dtype=tf.float32, shape=None):
        """
        :param op_name:
        :param arrary:
        :param dtype:
        :param shape:
        :return:
        """
        new_node = tf.NodeDef()
        new_node.op = 'Const'
        new_node.name = op_name
        new_node.attr['dtype'].CopyFrom(tf.AttrValue(type=dtype.as_datatype_enum))
        assert list(np.shape(arrary)) == shape, "Please check the value"
        new_node.attr['value'].CopyFrom(
            tf.AttrValue(tensor=tf.make_tensor_proto(arrary, dtype, shape)))
        new_node.attr['_output_shapes'].CopyFrom(
            tf.AttrValue(list=tf.AttrValue.ListValue(shape=[tensor_shape_pb2.TensorShapeProto(
                dim=[tensor_shape_pb2.TensorShapeProto.Dim(size=x) for x in shape])])))
        return new_node

    @staticmethod
    def creat_relu_node(op_name, dtype=tf.float32):
        new_node = tf.NodeDef()
        new_node.op = 'Relu'
        new_node.name = op_name
        new_node.attr["T"].CopyFrom(tf.AttrValue(type=dtype.as_datatype_enum))
        return new_node


if __name__ == "__main__":

    graph = GraphEdit(graph_pb='./test_add.pb', input_node=['input'], output_node=['vgg_16/fc8/squeezed'])
    print(graph.search_node('vgg_16/conv1/conv1_1/Conv2D'))
    #
    # new_graph = graph.delete_node(node_name='vgg_16/conv1/conv1_1/Relu')
    # graph.save(['vgg_16/fc8/squeezed'], './test_remove.pb')
