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
python generate_data_stdout.py "$NUM_SAMPLES" | hdfs dfs -put - "$HDFS_DATA_PATH"
```
