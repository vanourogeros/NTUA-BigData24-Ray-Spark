# NTUA-BigData24-Ray-Spark

## Authors
- Theodoros Papadopoulos  03119017 [paptheop01](https://github.com/paptheop01)
- Chrysa Pratikaki        03119131 [cpratikaki](https://github.com/cpratikaki)
- Ioannis Protogeros      03119008 [vanourogeros](https://github.com/vanourogeros)

## Requirements 
>ray==2.9.0 <br>
>pyspark==3.3.2 <br>
>torch==2.1.2 <br>

## HDFS Installation 

https://colab.research.google.com/drive/1pjf3Q6T-Ak2gXzbgoPpvMdfOHd1GqHZG?usp=sharing

## Spark Setup 

After the HDFS installation, simply run `start-dfs`, `start-yarn.sh` and `$SPARK_HOME/sbin/start-history-server.sh` on the head node to setup for the Spark scripts execution.

## Ray Setup 

Starting the cluster on the head node:

`RAY_GRAFANA_IFRAME_HOST=http://[public-ip]:3000 CLASSPATH=``$HADOOP_HOME/bin/hdfs classpath --glob`` ray start --head --node-ip-address=[head-node-address] --port=6379 --dashboard-host=0.0.0.0`

Connecting the two workers:

`CLASSPATH=``$HADOOP_HOME/bin/hdfs classpath --glob`` ray start --address=[head-node-address]`
