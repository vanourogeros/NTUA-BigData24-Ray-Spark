The scripts here perform triangle counting on a graph that is saved in an edge-list TSV file in the hdfs: `hdfs://okeanos-master:54310/graphs/web-Google.txt`

Spark uses GraphX (Graphframes), and Ray parallelizes the `triangles` function from NetworkX. 
