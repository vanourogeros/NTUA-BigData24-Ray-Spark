from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from graphframes import *
from sparkmeasure import TaskMetrics
import os
import sys
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

# Initialize Spark
sparky = SparkSession \
    .builder \
    .appName("triangle counting") \
    .master("yarn") \
    .config("spark.jars.packages", "ch.cern.sparkmeasure:spark-measure_2.12:0.23,graphframes:graphframes:0.8.3-spark3.5-s_2.12") \
    .getOrCreate() \

sc = sparky.sparkContext

taskmetrics = TaskMetrics(sparky)

# Define a simple graph (replace this with your graph data)
vertices = sparky.createDataFrame([
    (1, "A"),
    (2, "B"),
    (3, "C"),
    (4, "D"),
    (5, "E")
], ["id", "name"])

edges = sparky.createDataFrame([
    (1, 2),
    (2, 3),
    (3, 1),
    (3, 4),
    (4, 5),
    (5, 3)
], ["src", "dst"])

print(edges)
# Create a GraphFrame
graph = GraphFrame(vertices, edges)

# Triangle counting using GraphFrames

taskmetrics.begin()

triangles = graph.triangleCount().show()

taskmetrics.end()
taskmetrics.print_report()

# Display the triangle count
#triangles.show()

# Stop Spark
sc.stop()
