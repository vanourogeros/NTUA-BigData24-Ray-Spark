import ray
from pyarrow import fs
import sys
import torch.nn as nn
from ray.data.preprocessors import MinMaxScaler, Concatenator
import torch
import torch.optim as optim
from ray.train import ScalingConfig
from ray import train, tune
from ray.train.torch import TorchTrainer
import time
from ray.train import Checkpoint
from ray.tune.schedulers import ASHAScheduler

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.sigmoid(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x
    

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

def train_func(config):

    epochs = config["epochs"]

    batch_size = config["batch_size"]
    input_size = config["input_size"]
    hidden_size = config["hidden_size"]

    model = SimpleNN(input_size, hidden_size)
    model = ray.train.torch.prepare_model(model)
    criterion = nn.BCELoss()  # Binary Cross Entropy Loss for binary classification
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_data_shard = train.get_dataset_shard("train")
    train_dataloader = train_data_shard.iter_torch_batches(
        batch_size=batch_size, dtypes=torch.float32
    )
    val_data_shard = train.get_dataset_shard("val")
    val_dataloader = val_data_shard.iter_torch_batches(
        batch_size=batch_size, dtypes=torch.float32
    )

    for epoch_idx in range(epochs):
        model.train()
        for batch in train_dataloader:
            inputs, labels = batch["features"], batch["label"]
             # Forward pass
            outputs = model(inputs).squeeze()

            # Convert labels to float for BCELoss
            labels = labels.float().squeeze()

            # Calculate loss
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        model.eval()
        test_loss, num_correct, num_total = 0, 0, 0
        with torch.no_grad():
            for batch in val_dataloader:
                inputs, labels = batch["features"], batch["label"]
                # Forward pass
                outputs = model(inputs).squeeze()
                labels = labels.float().squeeze()
                loss = criterion(outputs, labels)

                test_loss += loss.item()
                num_total += labels.shape[0]
                num_correct += ((outputs > 0.5).float() == labels).sum().item()
            accuracy = num_correct / num_total
            print("="*100,accuracy)
            print("="*100,loss)
        ray.train.report(metrics={"accuracy": accuracy, "loss":loss})



# trainer = TorchTrainer(
#     train_func,
#     datasets={"train": train_ds, "val": val_ds},
#     train_loop_config={"epochs": 10},
#     scaling_config=ScalingConfig(num_workers=3, use_gpu=False)
# )

# result = trainer.fit()
# print(f"Training result: {result}")

# total_time = time.time() - start_time

# print("Preprocessing Time:", preprocessing_time)
# print("Total Time", total_time)
# print(ds.stats())


def main(num_samples=2, max_num_epochs=10):
    config = {
        "epochs": 10,
        "batch_size": 512,#[512, 1024],
        "hidden_size": [128, 256],
        "input_size": 4#[4 , 8]
    }
    scheduler = ASHAScheduler(
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train_func),
            resources={"cpu": 4, "gpu": 0}
        ),
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            scheduler=scheduler,
            num_samples=num_samples,
        ),
        param_space=config,
    )
    results = tuner.fit()
    
    best_result = results.get_best_result("loss", "min")
    print("Best trial config: {}".format(best_result.config))
    print("Best trial final validation loss: {}".format(
        best_result.metrics["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_result.metrics["accuracy"]))

main(num_samples=2, max_num_epochs=10)
