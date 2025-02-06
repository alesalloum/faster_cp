import networkx as nx
import random

import matplotlib.pyplot as plt

import faster_cp


def generate_core_periphery_graph(n_core, n_periphery, p_core=0.8, p_periphery=0.1, p_cross=0.3):
    """
    Generates a core-periphery structured graph using networkx with integer node names.

    Parameters:
        n_core (int): Number of core nodes.
        n_periphery (int): Number of periphery nodes.
        p_core (float): Probability of edges within the core.
        p_periphery (float): Probability of edges within the periphery.
        p_cross (float): Probability of edges between core and periphery.

    Returns:
        G (networkx.Graph): A graph with a core-periphery structure.
    """
    G = nx.Graph()

    # Assign integer node IDs
    core_nodes = list(range(n_core))
    periphery_nodes = list(range(n_core, n_core + n_periphery))

    G.add_nodes_from(core_nodes, type="core")
    G.add_nodes_from(periphery_nodes, type="periphery")

    # Connect core nodes (densely)
    for i in core_nodes:
        for j in core_nodes:
            if i < j and random.random() < p_core:
                G.add_edge(i, j)

    # Connect periphery nodes (sparsely)
    for i in periphery_nodes:
        for j in periphery_nodes:
            if i < j and random.random() < p_periphery:
                G.add_edge(i, j)

    # Connect periphery nodes to core (moderate connections)
    for periphery in periphery_nodes:
        for core in core_nodes:
            if random.random() < p_cross:
                G.add_edge(periphery, core)

    return G

# Example usage
if __name__ == "__main__":
    G = generate_core_periphery_graph(n_core=100, n_periphery=500)
    
    #print("Edges:", list(G.edges()))  # Print edges as tuples
    # Draw the network with different colors
    core_nodes = [node for node, data in G.nodes(data=True) if data["type"] == "core"]
    periphery_nodes = [node for node, data in G.nodes(data=True) if data["type"] == "periphery"]

    pos = nx.spring_layout(G)  # Position nodes
    nx.draw(G, pos, with_labels=True, node_size=500)
    nx.draw_networkx_nodes(G, pos, nodelist=core_nodes, node_color="red", label="Core")
    nx.draw_networkx_nodes(G, pos, nodelist=periphery_nodes, node_color="blue", label="Periphery")
    plt.legend()

    # Save the plot instead of displaying it
    plt.savefig("core_periphery_graph.png", dpi=300, bbox_inches="tight")  # Save with high resolution
    plt.close()  # Close the plot to free memory
    
    ### RUN INFERENCE

    g = faster_cp.GraphWrapper()
    #g.add_edges([(0, 1), (1, 2), (2, 3), (3, 0), (1, 3)]
    #g.add_edges([(10, 20), (20, 30), (30, 40), (40, 10), (20, 40)])
    edges_to_add = G.edges()
    #print(edges_to_add)
    g.add_edges(edges_to_add)

    #print("Nodes:", g.node_count())  # Output: 4
    #print("Edges:", g.edge_count())  # Output: 5

    init_partition = {node: random.choice([0, 1]) for node in G.nodes()}
    g.set_partition(init_partition)

    best_partition = faster_cp.simulated_annealing_partition(
        g, max_iter=10000, initial_temp=1.0, cooling_rate=0.999
    )

    print("Best partition:", best_partition)
    #print("Best DL:", best_dl)





    

G = generate_core_periphery_graph(n_core=10, n_periphery=20)




