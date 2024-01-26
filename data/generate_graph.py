import networkx as nx
import sys

# Create a Watts-Strogatz model with initial REG degree of 100, nodes=(given), and p=0.1
G = nx.watts_strogatz_graph(int(sys.argv[1]), 100, 0.1).to_directed()

# Save the graph as a TSV file
nx.write_edgelist(G, "graph.tsv", delimiter="\t", data=False)
