import pandas as pd
import pyarrow as pa
import ray
import numpy as np
from ray.data import Dataset, from_arrow, from_pandas, from_items
from ray.data.aggregate import AggregateFn
from functools import partial
from typing import Iterable, Tuple


def computeContribs(row):
    urls = row['ToNodes']
    if type(urls) == int: # In some environments if there is only one node in row['ToNodes'] this is treated as an int
        urls = [urls]
    print(urls)
    rank = row['PageRank']
    """Calculates URL contributions to the rank of other URLs."""
    num_urls = len(urls)
    for url in urls:
        yield {'item' : (url, rank / num_urls)}

#ray.init()

# Create a Ray Dataset
ds = ray.data.from_pandas(
   pd.DataFrame(
       {
           "FromNode": np.random.randint(1, 100, 1000) ,
           "ToNode": np.random.randint(1, 100, 1000) ,
       }
   )
)

def find_incoming_edges(group):
    group["FromNode"] = list(group["FromNode"])
    return group

# Needed in some environments, sometimes not needed
ctx = ray.data.context.DataContext.get_current()
ctx.enable_tensor_extension_casting = False


# Transformations to extract nodes from dataset
nodes = ray.data.from_items(ds.select_columns(["FromNode"])
          .union(ds.select_columns(["ToNode"]).add_column("FromNode", lambda df: df["ToNode"]))
          .unique("FromNode")).add_column("Node", lambda df: df["item"]).drop_columns(["item"]) \
          .sort("Node")
print(nodes.schema())

N = nodes.count()

data = nodes.zip(ds.groupby("FromNode").count().sort("FromNode")).drop_columns(["FromNode"])

data.show()



print(ds.groupby("ToNode"))

# Aggregation function to map edges grouped by FromNode to list of all the Nodes
# they point to (crucial element of PageRank algorithm)
aggregation = AggregateFn(
    init=lambda column: np.array([], dtype=int),
    accumulate_row=lambda a, row: np.append(a, row["ToNode"]),
    merge = lambda a1, a2: np.append(a1,a2),
    name="ToNodes"
)
incoming_edges_ds = ds.groupby("FromNode").aggregate(aggregation).sort("FromNode")

data = data.zip(incoming_edges_ds.drop_columns("FromNode"))

# Initialize PageRank values
data = data.add_column("PageRank", lambda df: 1)

alpha = 0.85
num_iters = 10

data.show()

# splits [x1,x2] values in two columns x1,x2
def split_item(batch):
    items = np.array(batch["item"].tolist())
    batch["Node"] = items[:, 0]
    batch["contrib"] = items[:, 1]
    del batch["item"]
    return batch

# pagerank score calculation
def map_contrib(batch):
  batch['sum(contrib)'] = 0.15 + 0.85 * batch['sum(contrib)']
  return batch

# dataset transformation for new PageRank iteration
def new_iter(batch):
  batch["PageRank"] = batch["sum(contrib)"]
  del batch["sum(contrib)"], batch["Node_1"]
  return batch

num_iters = 10
for i in range(num_iters):
  contributions = data.flat_map(computeContribs) \
                      .map_batches(split_item) \
                      .groupby("Node").sum("contrib") \
                      .map_batches(map_contrib)

  # This is unfortunate, but necessary because of Ray's (v. 2.9.0) strict restrictions
  # when it comes to the dataset.zip() operation, regarding the underlying block types
  # see https://github.com/ray-project/ray/issues/31550: This problem is planned to be
  # adressed with Ray v. 2.10.0 release.
  # data = contributions.zip(data).map_batches(new_iter)
  data = from_pandas(data.to_pandas()).zip(from_pandas(contributions.to_pandas())).map_batches(new_iter)

data.show()
