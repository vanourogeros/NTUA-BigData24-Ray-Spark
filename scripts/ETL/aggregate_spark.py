from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from sparkmeasure import TaskMetrics, StageMetrics
from pyspark.sql.functions import col, sum, count
import os
import sys
import time
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

# Initialize Spark
sparky = SparkSession \
    .builder \
    .appName("aggregate_data") \
    .master("yarn") \
    .config("spark.executor.instances", sys.argv[1]) \
    .config("spark.jars.packages", "ch.cern.sparkmeasure:spark-measure_2.12:0.23") \
    .getOrCreate()

sc = sparky.sparkContext

stagemetrics = StageMetrics(sparky)

#df = spark.read.csv("hdfs://okeanos-master:54310/data/generated_data.csv", header=True, inferSchema=True)
df = sparky.read.format("csv") \
           .option("header", "true") \
           .option("inferSchema", "true") \
           .load("hdfs://okeanos-master:54310/data/generated_data.csv")


stagemetrics.begin()

# GroupBy and Aggregate
df_grouped = (
    df
    .groupBy("categorical_feature_1")
    .agg(
        sum("feature_3").alias("sum_feature_3")
    )
)

# Display the grouped and aggregated data
print("\nGrouped and Aggregated Data Rows:")
print(df_grouped.count())

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
print("memory report never ready :(")
# Stop Spark
sc.stop()
