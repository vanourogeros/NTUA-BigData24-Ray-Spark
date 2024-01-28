In here there are various scripts that perform training on the dataset that is generated with the script in the data folder of this GitHub project.
There is also a pipeline that preprocesses the data with Spark and writes them in parquet form in the HDFS, and then Ray reads the data from the HDFS, performs basic last-mile preprocessing, 
and then trains a PyTorch MLP classifier.
