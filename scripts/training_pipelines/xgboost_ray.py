import ray
from ray.train import ScalingConfig
from ray.train.xgboost import XGBoostTrainer
from pyarrow import fs
import sys
import torch.nn as nn
from ray.data.preprocessors import MinMaxScaler, Concatenator
import torch
import torch.optim as optim
from ray.train import ScalingConfig
import time
from sklearn.preprocessing import LabelEncoder

def filter_function(batch):
    batch = batch.drop(batch[batch["new_feature"] < batch["word"].str.len()].index)
    return batch
# Load data.
ray.init()

start_time = time.time()

hdfs_fs = fs.HadoopFileSystem.from_uri("hdfs://okeanos-master:54310")
ds = ray.data.read_csv(sys.argv[1], filesystem=hdfs_fs) \
        #.map_batches(lambda batch: batch)

ds = ds.add_column("new_feature", lambda df: df["feature_1"] ** 2 + df["feature_2"]**2) \
        .map_batches(filter_function, batch_format="pandas") \
        .select_columns(["feature_1", "feature_2", "feature_3", "new_feature", "label"])


encoder = LabelEncoder()
ds = ds.map_batches(lambda df: df.assign(word=encoder.fit_transform(df["word"])), batch_format="pandas")

ds = ds.map_batches(lambda df: df.astype({"categorical_feature_1": "category", "categorical_feature_2": "category", "word": "category"}), batch_format="pandas")
# Preprocess the data for training
preprocessor = Concatenator(output_column_name="features", exclude=["label"])

scaler = MinMaxScaler(["feature_1", "feature_2", "feature_3", "new_feature"])
ds = scaler.fit_transform(ds)
ds = preprocessor.fit_transform(ds)



print(ds.stats())
train_ds, val_ds = ds.train_test_split(0.3)

preprocessing_time = time.time() - start_time

trainer = XGBoostTrainer(
    scaling_config=ScalingConfig(
        # Number of workers to use for data parallelism.
        num_workers=3,
        # Whether to use GPU acceleration. Set to True to schedule GPU workers.
        use_gpu=False,
    ),
    label_column="label",
    num_boost_round=20,
    params={
        # XGBoost specific params (see the `xgboost.train` API reference)
        "objective": "binary:logistic",
        # uncomment this and set `use_gpu=True` to use GPU for training
        # "tree_method": "gpu_hist",
        "eval_metric": ["logloss", "error"],
        "enable_categorical": True,
    },
    datasets={"train": train_ds, "valid": val_ds},
    # If running in a multi-node cluster, this is where you
    # should configure the run's persistent storage that is accessible
    # across all worker nodes.
    # run_config=ray.train.RunConfig(storage_path="s3://..."),
)
result = trainer.fit()
print(result.metrics)
