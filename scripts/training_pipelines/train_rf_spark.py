from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.sql import SparkSession
from pyspark import SparkConf, SparkContext
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from sparkmeasure import TaskMetrics, StageMetrics
from pyspark.sql.functions import col, sqrt, length
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.torch.distributor import TorchDistributor
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
import os
import sys
import time

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable


# Initialize Spark
sparky = SparkSession \
    .builder \
    .appName("xgboost_spark") \
    .master("yarn") \
    .config("spark.executor.instances", sys.argv[1])\
    .config("spark.executor.cores", "4") \
    .config("spark.jars.packages", "ch.cern.sparkmeasure:spark-measure_2.12:0.23") \
    .getOrCreate() 

sc = sparky.sparkContext

stagemetrics = StageMetrics(sparky)
stagemetrics.begin()

#df = spark.read.csv("hdfs://okeanos-master:54310/data/generated_data.csv", header=True, inferSchema=True)
df = sparky.read.format("csv") \
           .option("header", "true") \
           .option("inferSchema", "true") \
           .load([f"hdfs://okeanos-master:54310/data/large/dataset{i}.csv" for i in [1,7]])

start_time = time.time()

df = df.withColumn(
    "new_feature",
    sqrt(col("feature_1") ** 2 + col("feature_2") ** 2)
).filter(length(col("word")) > col("new_feature"))
 
df = df.select(df.feature_1, df.feature_2, df.feature_3, df.new_feature, df.label)

assembler = VectorAssembler(inputCols=["feature_1", "feature_2", "feature_3", "new_feature"],
                             outputCol="input")
scaler = MinMaxScaler(inputCol="input", outputCol="features")
pipeline = Pipeline(stages=[assembler, scaler])
scalerModel = pipeline.fit(df)
scaledData = scalerModel.transform(df).select(["features", "label"])

#scaledData.show()

trainingData, testData = scaledData.randomSplit(weights= [0.7,0.3], seed=100)
#trainingData.show()
preprocessing_time = time.time() - start_time
# Train a GBT model.
print("Preproccessing Time:", preprocessing_time)
rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=5)

# Chain indexers and GBT in a Pipeline
#labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel",
                            #labels=labelIndexer.labels)

# Chain indexers and forest in a Pipeline
pipeline = Pipeline(stages=[ rf])

# Train model.  This also runs the indexers.
model = pipeline.fit(trainingData)

# Make predictions.
predictions = model.transform(testData)

# Select example rows to display.
#predictions.select("prediction", "label", "features").show(5)

# Select (prediction, true label) and compute test error
evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test Error = %g" % (1.0 - accuracy))

gbtModel = model.stages[0]
print(gbtModel)  # summary only
# $example off$
total_time = time.time() - start_time
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

print("Preproccessing Time:", preprocessing_time)
print("Total Time:", total_time)

