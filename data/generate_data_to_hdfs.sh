#!/bin/bash

# This script executes the python script that generates
# classification data with number of samples equal to $1 (default:5000)
# and writes a csv file to the hdfs path specified (default:/data/generated_data.csv)
# with hdfs dfs -put. Obviously you need to have and hdfs configured and set up

NUM_SAMPLES=${1:-5000}
HDFS_DATA_PATH=${2:-/data/generated_data.csv}
python generate_data_stdout.py "$NUM_SAMPLES" | hdfs dfs -put - "$HDFS_DATA_PATH"
