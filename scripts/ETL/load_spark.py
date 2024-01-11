from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from sparkmeasure import TaskMetrics, StageMetrics
import os
import sys
import time
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

# Initialize Spark
sparky = SparkSession \
    .builder \
    .appName("load_data") \
    .master("yarn") \
    .config("spark.jars.packages", "ch.cern.sparkmeasure:spark-measure_2.12:0.23") \
    .getOrCreate() \

sc = sparky.sparkContext

stagemetrics = StageMetrics(sparky)
stagemetrics.begin()

#df = spark.read.csv("hdfs://okeanos-master:54310/data/generated_data.csv", header=True, inferSchema=True)
df = sparky.read.format("csv") \
           .option("header", "true") \
           .option("inferSchema", "true") \
           .load("hdfs://okeanos-master:54310/data/generated_data.csv").collect()

stagemetrics.end()
stagemetrics.print_report()
print(stagemetrics.aggregate_stagemetrics())

# memory report needs a bit of time to run...
for _ in range(20):
    try:
        stagemetrics.print_memory_report()
    except:
        print("memory report not ready")
        time.sleep(1)
print("memory report never ready :(")
#print(stagemetrics.aggregate_stagemetrics())
# Display the triangle count
#triangles.show()

# Stop Spark
sc.stop()
