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
This is an example implementation of PageRank. For more conventional use,
Please refer to PageRank implementation provided by graphx
"""
import re
import sys
from operator import add
from typing import Iterable, Tuple
import time
from pyspark.resultiterable import ResultIterable
from pyspark.sql import SparkSession
from sparkmeasure import TaskMetrics, StageMetrics
import os, sys

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

def computeContribs(urls: ResultIterable[str], rank: float) -> Iterable[Tuple[str, float]]:
    """Calculates URL contributions to the rank of other URLs."""
    num_urls = len(urls)
    for url in urls:
        yield (url, rank / num_urls)


def parseNeighbors(urls: str) -> Tuple[str, str]:
    """Parses a urls pair string into urls pair."""
    parts = re.split(r'\s+', urls)
    return parts[0], parts[1]


if __name__ == "__main__":

    spark = SparkSession \
        .builder \
        .appName("PageRank") \
        .master("yarn") \
        .config("spark.executor.instances", sys.argv[1]) \
        .config("spark.executor.cores", "4") \
        .config("spark.jars.packages", "ch.cern.sparkmeasure:spark-measure_2.12:0.23,graphframes:graphframes:0.8.3-spark3.5-s_2.12") \
        .getOrCreate()

    #if len(sys.argv) != 3:
     #   print("Usage: pagerank <file> <iterations>", file=sys.stderr)
      #  sys.exit(-1)

    print("WARN: This is a naive implementation of PageRank and is given as an example!\n" +
          "Please refer to PageRank implementation provided by graphx",
          file=sys.stderr)

    stagemetrics = StageMetrics(spark)
    stagemetrics.begin()
    # Loads in input file. It should be in format of:
    #     URL         neighbor URL
    #     URL         neighbor URL
    #     URL         neighbor URL
    #     ...i



    lines = spark.read.text("hdfs://okeanos-master:54310/graphs/web-Google.txt").rdd.map(lambda r: r[0])

    # Loads all URLs from input file and initialize their neighbors.
    links = lines.map(lambda urls: parseNeighbors(urls)).distinct().groupByKey().cache()

    # Loads all URLs with other URL(s) link to from input file and initialize ranks of them to one.
    ranks = links.map(lambda url_neighbors: (url_neighbors[0], 1.0))

    # Calculates and updates URL ranks continuously using PageRank algorithm.
    for iteration in range(10):
        # Calculates URL contributions to the rank of other URLs.
        contribs = links.join(ranks).flatMap(lambda url_urls_rank: computeContribs(
            url_urls_rank[1][0], url_urls_rank[1][1]  # type: ignore[arg-type]
        ))

        # Re-calculates URL ranks based on neighbor contributions.
        ranks = contribs.reduceByKey(add).mapValues(lambda rank: rank * 0.85 + 0.15)

    # Collects all URL ranks and dump them to console.
    print(ranks.take(20))
    stagemetrics.end()
    stagemetrics.print_report()

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
    #for (link, rank) in ranks.collect():
    #    print("%s has rank: %s." % (link, rank))

    spark.stop()
