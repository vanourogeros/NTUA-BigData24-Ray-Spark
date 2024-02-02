## Classification Data Generation

In here there is a script that generates a classification dataset CSV (with classification-relevant numerical features and some random categorical features for measuring ETL operations performance) of desired size directly into a set up HDFS (useful when we want to generate data that doesn't fit into the disk of a single machine). Use it with

```bash
chmod +x generate_data_to_hdfs.sh
./generate_data_to_hdfs.sh <num_samples> </path/in/hdfs.csv>
```
e.g:
```bash
./generate_data_to_hdfs.sh 100000 /data/generated_data.csv
```

The python script writes to stdout in the form of a stream, so the bash script redirects that output into the hdfs file with the command
```bash
# "$NUM_SAMPLES" and "$HDFS_DATA_PATH" are given by the user
python generate_data_stdout.py "$NUM_SAMPLES" | hdfs dfs -put - "$HDFS_DATA_PATH"
```

## Graph Data Retrieval and Generation

The real-world graphs that were used for our experiments were found in the KONECT Project site  (http://konect.cc/). Example of downloading a TSV of a graph and putting it into the HDFS:
```bash
wget http://konect.cc/files/download.tsv.higgs-twitter-social.tar.bz2
tar -xvf download.tsv.higgs-twitter-social.tar.bz2
hdfs dfs -put /higgs-twitter-social/out.higgs-twitter-social /graphs/higgs.txt
```

We also used generated Small-World graphs using the Watts-Strogatz model with the  `generate_graph.py` scripts which uses the NetworkX library. Usage:
```bash
python generate_graph.py 10000 # Generate a small world graph with 10000 nodes (it will also have 1.000.000 edges)
hdfs dfs -put graph.tsv /graphs/10000.txt
```

## Speech Emotion Recognition (RAVDESS Dataset)

Kaggle API needs to be installed
```bash
pip install kaggle
```

And an API token from your kaggle account put in `/.kaggle/kaggle.json`. Then you can download the dataset and unzip it with
```bash
kaggle datasets download uwrfkaggler/ravdess-emotional-speech-audio –path ./data/ravdess –unzip
```
Then put the dataset in the HDFS with `hdfs -put` for it to be accessible from all nodes.
