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
from xgboost.spark import SparkXGBClassifier
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
    .config("spark.executor.instances", "3") \
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

scaledData = scaledData.select(scaledData.features, scaledData.label)
# Rename the sliced_features column to feature_1, feature_2, feature_3



train_df, val_df = scaledData.randomSplit (weights= [0.7,0.3], seed=100)

preprocessing_time = time.time() - start_time


layers = [4, 128, 2]
# create the trainer and set its parameters
trainer = SparkXGBClassifier(
  features_col="features",
  label_col="label",
  num_workers=3,
)
model = trainer.fit(train_df)
result = model.transform(val_df)
predictionAndLabels = result.select("prediction", "label")
evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
print("Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))

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
model.save("/tmp/xgboost-pyspark-model")

sc.stop()

print("Preproccessing Time:", preprocessing_time)
print("Total Time:", total_time)
