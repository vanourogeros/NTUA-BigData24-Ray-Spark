Various scripts for testing ETL operations with Spark and Ray. All scripts suppose that all cluster nodes can access an HDFS in `hdfs://okeanos-master:54310` (Change if needed).
Running the scripts is a matter of simply calling them with python. Ray scripts need the path of the data inside the HDFS and Spark scripts need the number of executors and the path in the HDFS. e.g:
We suppose we have a folder with multiple CSVs (with the schema that the script in the data directory of this GitHub project generates) in a directory /data/classification in the HDFS

``bash
python transform_ray.py /data/classification
``

``bash
python transform_spark.py 3 /data/classification
``

For sort_ray.py, we also set the env variable that is shown to enable push-based shuffle, since we figured it improves performance in the sorting task (this is still an experimental feature of Ray):
``bash
RAY_DATA_PUSH_BASED_SHUFFLE=1 python sort_ray.py /data/classification
``
More info: https://docs.ray.io/en/latest/data/performance-tips.html#enabling-push-based-shuffle
