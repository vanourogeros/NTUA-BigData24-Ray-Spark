import ray
import networkx as nx
from pyarrow import fs
import sys
import subprocess
import time

cat = subprocess.Popen(["hadoop", "fs", "-cat", "/graphs/web-Google.txt"], stdout=subprocess.PIPE)
lines = [line.decode() for line in cat.stdout if not line.startswith('#'.encode())]
print(lines[0])

num_chunks = int(sys.argv[1])

G = nx.parse_edgelist(lines, nodetype=int, create_using=nx.DiGraph())
G = G.to_undirected()
node_chunk_size = len(G.nodes()) // int(sys.argv[1])
node_list = list(G.nodes())
N = node_chunk_size

node_chunks = [
                node_list[N*i : N*(i+1)] if i < N - 1
                else node_list[N*i:]
                for i in range(num_chunks)
                ]


ray.init()

@ray.remote
def compute_triangles(G, nodes):
    return nx.triangles(G, nodes=nodes)

A = time.time()
G_reference = ray.put(G)

#print(node_chunks[1])

results = [ray.get(compute_triangles.remote(G_reference, nodes=node_chunks[i])) for i in range(num_chunks)]
#triangles_first_half = ray.get(compute_triangles.remote(G_reference, first_half_nodes))
#triangles_second_half = ray.get(compute_triangles.remote(G_reference, second_half_nodes))



print(f"triangles in the first half of the graph: {results[0]}")
print(f"triangles in the second half of the graph: {results[1]}")
print("Time:", time.time() - A)