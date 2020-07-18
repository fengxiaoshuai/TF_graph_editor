# OptimizerTooll

# About

Optimizer Tooll is a frequently used tool for optimizing neural networks   
GraphEdit.py : a tool to modify Graph of tensorflow 

# install
The appropriate files are downloaded to the local machine

# Run
--------------------------------------------------------------------------
Build node

for example:     
    import GraphEdit as ge  
    ge.BuildNdoe().creat_conv_node()  

--------------------------------------------------------------------------
Edit graph

for example:   
	import GraphEdit as ge   
	old_graph = './test.pb'    
	graph_edit = ge.GraphEdit(graph_pb=old_graph, input_node=['input'], output_node=['vgg_16/fc8/squeezed'])   
	print(graph_edit)

