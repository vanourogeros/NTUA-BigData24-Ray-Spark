import ray
from pyarrow import fs
import sys


ray.init()

ctx = ray.data.DataContext.get_current()
ctx.use_push_based_shuffle = True
ctx.execution_options.resource_limits.cpu = int(sys.argv[1])

hdfs_fs = fs.HadoopFileSystem.from_uri("hdfs://okeanos-master:54310")

ds = ray.data.read_csv([f"/data/large/dataset{i}.csv" for i in range(1,sys.argv[2])], filesystem=hdfs_fs) \
       .map_batches(lambda batch: batch)

ds_sorted = ds.sort("feature_1")
ds_sorted.show()
