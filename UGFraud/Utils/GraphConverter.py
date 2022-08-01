import networkx as nx
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_networkx

class GraphConverter(object):
	@classmethod
	def frame_pubmed(cls):
		geometric_graph 	= Planetoid(root='/tmp/pubmed', name='Pubmed')[0]
		
		nx_graph 			= to_networkx(geometric_graph, node_attrs = ["val_mask", "y"])
		max_node_id 		= max(nx_graph.nodes())
		unknown_class_label = geometric_graph.y.min().item()

		nx.set_node_attributes(nx_graph, 0.0, name = 'prior')
		nx.set_node_attributes(nx_graph, 'user', 'types')

		for each_node in nx_graph.nodes():
			label 	= nx_graph.nodes()[each_node]["y"] +1
			mask 	= nx_graph.nodes()[each_node]["val_mask"]
			nx_graph.nodes()[each_node]["label"] = label if not mask else unknown_class_label

		for (left, right) in list(nx_graph.edges()):
			max_node_id += 1
			nx_graph.add_node(max_node_id, prior = 0.0, types = 'prod')
			nx_graph.remove_edge(left, right)
			nx_graph.add_edge(left, max_node_id)
			nx_graph.add_edge(right, max_node_id)

		nx.set_edge_attributes(nx_graph, 'review', 'types')
		nx.set_edge_attributes(nx_graph, 0, 'label')
		nx.set_edge_attributes(nx_graph, 0, 'prior')
		nx_graph = nx.relabel_nodes(nx_graph, {n : str(n) for n in nx_graph.nodes()})
		return nx_graph