import ray
from pyarrow import fs

ray.init()
hdfs_fs = fs.HadoopFileSystem.from_uri("hdfs://okeanos-master:54310")
ds = ray.data.read_csv("/data/generated_data.csv", filesystem=hdfs_fs)

print(ds.materialize().stats())
