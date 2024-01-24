from ml.dmlc.xgboost4j.scala.spark import XGBoostClassifier
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from sparkmeasure import TaskMetrics, StageMetrics
from pyspark.sql.functions import col, sqrt, length
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.torch.distributor import TorchDistributor
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit, CrossValidator


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
    .config("spark.executor.instances", "12") \
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
           .load("hdfs://okeanos-master:54310"+"/data/large")

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
scaledData = scalerModel.transform(df)

train_df, val_df = scaledData.randomSplit (weights= [0.7,0.3], seed=100)

preprocessing_time = time.time() - start_time
# Create the trainer and set its parameters
trainer = XGBoostClassifier(
    featuresCol="features",
    labelCol="label",
    predictionCol="prediction",
    numWorkers=3,  # Number of workers for distributed training
    numEarlyStoppingRounds=3,  # Early stopping rounds
    maxIter=10  # Maximum number of iterations
)

# Define the parameter grid
paramGrid = ParamGridBuilder() \
    .addGrid(trainer.maxDepth, [3, 5]) \
    .addGrid(trainer.eta, [0.01, 0.1]) \
    .build()

# Define the evaluator
evaluator = MulticlassClassificationEvaluator(metricName="accuracy")

# Define the cross-validator
tvs  = TrainValidationSplit(
    estimator=trainer,
    estimatorParamMaps=paramGrid,
    evaluator=evaluator,
    numFolds=1
)
tvModel = tvs.fit(train_df)
# Get the best model from the cross-validation
best_model = tvModel.bestModel

# Make predictions on the validation set using the best model
result = best_model.transform(val_df)
predictionAndLabels = result.select("prediction", "label")

# Evaluate the accuracy of the best model
print("Best test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))

# Print the best parameters found during cross-validation
print("Best Max Iterations:", best_model.getMaxIter())
# print("Best Block Size:", best_model.getBlockSize())

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
