import ray
from pyarrow import fs
import torch.nn as nn
from ray.data.preprocessors import MinMaxScaler, Concatenator
import torch
import torch.optim as optim
from ray.train import ScalingConfig
from ray import train
from ray.train.torch import TorchTrainer
import time

start_time = time.time()

hdfs_fs = fs.HadoopFileSystem.from_uri("hdfs://okeanos-master:54310")
train_ds = ray.data.read_parquet("/spark_data/train", filesystem=hdfs_fs) \
        #.map_batches(lambda batch: batch)

val_ds = ray.data.read_parquet("/spark_data/val", filesystem=hdfs_fs) \
        #.map_batches(lambda batch: batch)

read_time = time.time() - start_time

# Preprocess the data for training
preprocessor = Concatenator(output_column_name="features", exclude=["label"])

scaler = MinMaxScaler(["feature_1", "feature_2", "feature_3", "new_feature"])

train_ds = scaler.fit_transform(train_ds)
train_ds = preprocessor.fit_transform(train_ds)
val_ds = scaler.fit_transform(val_ds)
val_ds = preprocessor.fit_transform(val_ds)

preprocess_time = time.time() - start_time

print("Read time:", read_time)
print("time until final preproccess", preprocess_time)

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
    

def train_func(config):

    epochs = config["epochs"]

    batch_size = 1024
    input_size = 4
    hidden_size = 128
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
        ray.train.report(metrics={"accuracy": accuracy})


trainer = TorchTrainer(
    train_func,
    datasets={"train": train_ds, "val": val_ds},
    train_loop_config={"epochs": 10},
    scaling_config=ScalingConfig(num_workers=3, use_gpu=False)
)

result = trainer.fit()
print(f"Training result: {result}")

total_time = time.time() - start_time
