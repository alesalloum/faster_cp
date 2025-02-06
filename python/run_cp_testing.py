import random

import networkx as nx
import faster_cp

print("Reading the graph")
G = nx.Graph(nx.read_graphml("./data/graphs/IMMIGRATION_23.graphml"))
G = nx.convert_node_labels_to_integers(G, first_label=0, ordering='default', label_attribute="origina_index")

partition = nx.get_node_attributes(G, "PPSBM")

# Mapping for the labels: A -> 0 and B -> 1
label_map = {'A': 0, 'B': 1}

partition_relabeled = {k: label_map[v] for k, v in partition.items()}

print("Performing inference")
# Initialize the graph
g = faster_cp.GraphWrapper()

# Add edges to the graph
edges_to_add = G.edges()
g.add_edges(edges_to_add)

# Improve the initial partition later
init_partition = {node: random.choice([0, 1]) for node in G.nodes()}
g.set_partition(init_partition)

best_partition = faster_cp.simulated_annealing_partition(
    g, max_iter=220000, initial_temp=1.0, cooling_rate=0.999
)

# Set node attributes
nx.set_node_attributes(G, best_partition, "hierarchy")

print("Saving the graph")
# Save to GraphML file
nx.write_graphml(G, "./data/decomposed_graphs/IMM.graphml")

#print("Best partition:", best_partition)