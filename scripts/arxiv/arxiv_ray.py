import ray
from pyarrow import fs

ray.init()
hdfs_fs = fs.HadoopFileSystem.from_uri("hdfs://okeanos-master:54310")
ds = ray.data.read_csv("/data/Books_rating.csv", filesystem=hdfs_fs,ray_remote_args={"num_cpus": 2.25})
print('dpme')

ray.data.DataContext.get_current().execution_options.verbose_progress = True

mean = ds.groupby(["User_id"]).max(["Price"],ignore_nulls=True)

print(f"The mean values are:\n{mean}\n")

most_frequent_submitter = ds.groupby("User_id").count().sort("count()", descending=True).limit(1)

# Print the most frequent value
for row in most_frequent_submitter.iter_rows():
    print(most_frequent_submitter)

print(most_frequent_submitter.stats())
#print(ds.materialize().stats())
