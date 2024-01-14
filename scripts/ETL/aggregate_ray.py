import ray
from pyarrow import fs
import sys

ray.init()
hdfs_fs = fs.HadoopFileSystem.from_uri("hdfs://okeanos-master:54310")
ds = ray.data.read_csv(sys.argv[1], filesystem=hdfs_fs) \
        #.map_batches(lambda batch: batch)

ds = ds.groupby("categorical_feature_1") \
       .sum("feature_3")


ds.show()
print(ds.count())
print("stats:", ds.stats())
