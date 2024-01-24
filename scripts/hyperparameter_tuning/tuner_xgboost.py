import sklearn.datasets
import sklearn.metrics
from ray.tune.schedulers import ASHAScheduler
from ray import train, tune


def filter_function(batch):
    batch = batch.drop(batch[batch["new_feature"] < batch["word"].str.len()].index)
    return batch

ray.init()

start_time = time.time()

hdfs_fs = fs.HadoopFileSystem.from_uri("hdfs://okeanos-master:54310")
ds = ray.data.read_csv(sys.argv[1], filesystem=hdfs_fs) \
        #.map_batches(lambda batch: batch)

ds = ds.add_column("new_feature", lambda df: df["feature_1"] ** 2 + df["feature_2"]**2) \
        .map_batches(filter_function, batch_format="pandas") \
        .select_columns(["feature_1", "feature_2", "feature_3", "new_feature", "label"])


# Preprocess the data for training
preprocessor = Concatenator(output_column_name="features", exclude=["label"])

scaler = MinMaxScaler(["feature_1", "feature_2", "feature_3", "new_feature"])
ds = scaler.fit_transform(ds)
ds = preprocessor.fit_transform(ds)

print(ds.stats())

train_ds, val_ds = ds.train_test_split(0.3)

preprocessing_time = time.time() - start_time




def train(config):
    # Load dataset
    train_data_shard = train.get_dataset_shard("train")
 
    val_data_shard = train.get_dataset_shard("val")
    # Split into train and test set
    train_x,  train_y = train_data_shard.select_columns(["features"]),train_data_shard.select_columns(["label"])
    test_x,  test_y = val_data_shard.select_columns(["features"]),val_data_shard.select_columns(["label"])
    # Build input matrices for XGBoost
    train_set = xgb.DMatrix(train_x, label=train_y)
    test_set = xgb.DMatrix(test_x, label=test_y)
    # Train the classifier
    results = {}
    xgb.train(
        config,
        train_set,
        evals=[(test_set, "eval")],
        evals_result=results,
        verbose_eval=False,
    )
    # Return prediction accuracy
    accuracy = 1.0 - results["eval"]["error"][-1]
    train.report({"mean_accuracy": accuracy, "done": True})


if __name__ == "__main__":
    config = {
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "error"],
        "max_depth": tune.randint(1, 9),
        "min_child_weight": tune.choice([1, 2, 3]),
        "subsample": tune.uniform(0.5, 1.0),
        "eta": tune.loguniform(1e-4, 1e-1),
    }
       scheduler = ASHAScheduler(
        max_t=10, grace_period=1, reduction_factor=2  # 10 training iterations
    )

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train),
            resources={"cpu": 4, "gpu": 0}
        ),
        tune_config=tune.TuneConfig(
            metric="eval-logloss",
            mode="min",
            scheduler=scheduler,
            num_samples= 10,
        ),
        param_space=search_space,
        datasets={"train": train_ds, "valid": val_ds},
    )
    results = tuner.fit()
    best_result = results.get_best_result()
    accuracy = 1.0 - best_result.metrics["eval-error"]
    print(f"Best model parameters: {best_result.config}")
    print(f"Best model total accuracy: {accuracy:.4f}")