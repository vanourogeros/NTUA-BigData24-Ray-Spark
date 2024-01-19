#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
The K-means algorithm written from scratch against PySpark. In practice,
one may prefer to use the KMeans algorithm in ML, as shown in
examples/src/main/python/ml/kmeans_example.py.

This example requires NumPy (http://www.numpy.org/).
"""
import sys
from typing import List

import numpy as np
from pyspark.sql import SparkSession
import os
import time
from sparkmeasure import TaskMetrics, StageMetrics
from pyspark.sql.functions import col, sqrt, length
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

def parseVector(line: str) -> np.ndarray:
    try:
        return np.array([float(x) for x in line.split(',')[:3]])
    except:
        return


def closestPoint(p: np.ndarray, centers: List[np.ndarray]) -> int:
    bestIndex = 0
    closest = float("+inf")
    for i in range(len(centers)):
        tempDist = np.sum((p - centers[i]) ** 2)
        if tempDist < closest:
            closest = tempDist
            bestIndex = i
    return bestIndex


if __name__ == "__main__":

    #if len(sys.argv) != 4:
     #   print("Usage: kmeans <file> <k> <convergeDist>", file=sys.stderr)
      #  sys.exit(-1)

    print("""WARN: This is a naive implementation of KMeans Clustering and is given
       as an example! Please refer to examples/src/main/python/ml/kmeans_example.py for an
       example on how to use ML's KMeans implementation.""", file=sys.stderr)

    sparky = SparkSession \
    .builder \
    .appName("load_data") \
    .master("yarn") \
    .config("spark.executor.instances", "3") \
    .config("spark.executor.cores", "4") \
    .config("spark.jars.packages", "ch.cern.sparkmeasure:spark-measure_2.12:0.23") \
    .getOrCreate()

    sc = sparky.sparkContext

    stagemetrics = StageMetrics(sparky)
    #stagemetrics.begin()
    """
    data = sparky.read.format("csv") \
           .option("header", "true") \
           .option("inferSchema", "true") \
           .load("hdfs://okeanos-master:54310"+sys.argv[1]) \
           .limit(int(sys.argv[2]))
    data = data.withColumn(
    "new_feature",
    sqrt(col("feature_1") ** 2 + col("feature_2") ** 2)
    ).filter(length(col("word")) > col("new_feature"))

    data = data.select(data.feature_1, data.feature_2, data.feature_3, data.new_feature)

    assembler = VectorAssembler(inputCols=["feature_1", "feature_2", "feature_3", "new_feature"],
                             outputCol="features")
    pipeline = Pipeline(stages=[assembler])
    scalerModel = pipeline.fit(data)
    data = scalerModel.transform(data).rdd
    """
    lines = sparky.read.text('hdfs://okeanos-master:54310'+sys.argv[1]).rdd.map(lambda r: r[0])
    data = lines.map(parseVector).cache()
    data = sc.parallelize(data.take(int(sys.argv[2])))
    data = data.zipWithIndex().filter(lambda x: x[1] > 0).map(lambda x: x[0])
    #print(data.take(10))
    stagemetrics.begin()
    K = 2
    convergeDist = 0
    kPoints = data.takeSample(False, K, 1)
    tempDist = 1.0
    for i in range(10):
        closest = data.map(
            lambda p: (closestPoint(p, kPoints), (p, 1)))
        pointStats = closest.reduceByKey(
            lambda p1_c1, p2_c2: (p1_c1[0] + p2_c2[0], p1_c1[1] + p2_c2[1]))
        newPoints = pointStats.map(
            lambda st: (st[0], st[1][0] / st[1][1])).collect()

        tempDist = sum(np.sum((kPoints[iK] - p) ** 2) for (iK, p) in newPoints)

        for (iK, p) in newPoints:
            kPoints[iK] = p
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
        print("memory report never ready :(")
    print("Final centers: " + str(kPoints))

    sparky.stop()
