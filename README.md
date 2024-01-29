# A Comparative Study of Ray and Apache Spark in Big Data and Machine Learning

## Authors
- Theodoros Papadopoulos  03119017 [paptheop01](https://github.com/paptheop01)
- Chrysa Pratikaki        03119131 [cpratikaki](https://github.com/cpratikaki)
- Ioannis Protogeros      03119008 [vanourogeros](https://github.com/vanourogeros)

## What is this even
This project contains a series of scripts that tackle various tasks centred around Big Data and AI/ML operations by leveraging Ray and Apache Spark's distributed computing capabilities.
Tasks found in the scripts folder include (all implemented with both Ray and Spark)
- ETL Operations on big datasets (CSVs)
- Graph Operations (PageRank, Triangle Counting)
- ML Operations and Pipelines:
  - K-Means Clustering
  - Data preprocessing and training an MLP Classification model (Ray-Pytorch, Spark-SparkML)
  - Training pipeline for Random Forest Classifier (Ray-XGBoost, Spark-SparkML)
  - Hyperparameter Tuning for MLP classifiers (Ray Tune, Spark-SparkML)
  - Feature extraction from the RAVDESS speech emotion classification dataset (distributed handling of large amounts of audio data)

In the `data` folder there are scripts for the data generation and retrieval (for real-world data) process.

Our report contains the results of experiments on those scripts to test Ray and Spark's performance and scalability with workers and data.
For many experiments, we use datasets of sizes of over 2, 4, and 8GB on a cluster of 3 machines with 4 CPUs and 8GB RAM (Testing for cases where the dataset does not fit into main memory)

## Requirements
(additional to included packages by miniconda like pandas)
>ray==2.9.0 <br>
>pyspark==3.3.2 <br>
>torch==2.1.2 <br>
>xgboost==2.0.3 <br>
>networkx==3.2.1 <br>
>librosa==0.10.1 <br>
>kaggle==1.6.3 <br>

install with `pip install -r requirements.txt`

Install Ray's necessary packages with:

`pip install ray[core,data,train,tune]`

## HDFS and Apache Spark Installation 

To set up the Okeanos-Knossos Virtual Machines and install Apache Hadoop, YARN and Spark environment we followed the guide linked below:

https://colab.research.google.com/drive/1pjf3Q6T-Ak2gXzbgoPpvMdfOHd1GqHZG?usp=sharing

## HDFS, YARN Setup

After completing the HDFS installation, simply run `start-dfs`, `start-yarn.sh` and `$SPARK_HOME/sbin/start-history-server.sh` on the head node to set up for the Spark scripts execution.

## Ray Setup 

Starting a Ray head node - initializing a cluster: (This supposes a remote connection to the head node, hence the environment variable and dashboard host being on all interfaces (0.0.0.0))
```bash
RAY_GRAFANA_IFRAME_HOST=http://[public-ip]:3000 CLASSPATH=`$HADOOP_HOME/bin/hdfs classpath --glob` ray start --head --node-ip-address=[head-node-address] --port=6379 --dashboard-host=0.0.0.0
```

Starting a Ray instance on a worker machine -and attaching it to the cluster-:
```bash
CLASSPATH=`$HADOOP_HOME/bin/hdfs classpath --glob` ray start --address=[head-node-address]
```

Note: If you're planning to connect to a remote machine with SSH and using port forwarding, or if you're running locally, then you may opt for this command instead for Starting a Ray head node:
```bash
CLASSPATH=`$HADOOP_HOME/bin/hdfs classpath --glob` ray start --head --node-ip-address=[head-node-address] --port=6379
```

For configuring the Ray Dashboard and embedding Grafana and Prometheus for metrics and visualizations follow the instructions provided on the link: https://docs.ray.io/en/latest/cluster/configure-manage-dashboard.html

