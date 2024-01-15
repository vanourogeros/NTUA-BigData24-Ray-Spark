from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from sparkmeasure import TaskMetrics, StageMetrics
from pyspark.sql.functions import col, sqrt, length
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
    .config("spark.executor.instances", sys.argv[1]) \
    .config("spark.executor.cores", "1") \
    .config("spark.jars.packages", "ch.cern.sparkmeasure:spark-measure_2.12:0.23") \
    .getOrCreate() 

sc = sparky.sparkContext

stagemetrics = StageMetrics(sparky)
stagemetrics.begin()

#df = spark.read.csv("hdfs://okeanos-master:54310/data/generated_data.csv", header=True, inferSchema=True)
df = sparky.read.format("csv") \
           .option("header", "true") \
           .option("inferSchema", "true") \
           .load("hdfs://okeanos-master:54310"+sys.argv[2])

df = df.withColumn(
    "new_feature",
    sqrt(col("feature_1") ** 2 + col("feature_2") ** 2)
)

# Filtering rows based on the condition len(df['word']) > new_feature
filtered_df = df.filter(length(col("word")) > col("new_feature"))

# Show the resulting DataFrame
filtered_df.show()
print(filtered_df.count())

stagemetrics.end()
stagemetrics.print_report()
print(stagemetrics.aggregate_stagemetrics())

# memory report needs a bit of time to run...
patience = 20
while patience > 0:
    try:
        stagemetrics.print_memory_report()
        patience = -1
    except:
        print("memory report not ready")
        time.sleep(1)
        patience -= 1

if patience == 0:
    print("memory report was never ready :(")
# Stop Spark
sc.stop()
# Stop Spark
sc.stop()
