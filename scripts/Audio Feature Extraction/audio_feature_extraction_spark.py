from hdfs import InsecureClient
import pandas as pd
import numpy as np
import librosa
import sys
import warnings
import time
from pyarrow import fs

if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

def list_hdfs_files(hdfs_url, hdfs_directory):
    """
    Lists files in an HDFS directory.

    Args:
        hdfs_url (str): The HDFS URL (e.g., 'hdfs://localhost:9000').
        hdfs_directory (str): The path to the HDFS directory.

    Returns:
        list: A list of file names in the specified HDFS directory.
    """
    client = InsecureClient(hdfs_url)
    files = client.list(hdfs_directory)
    return files

# Example usage
hdfs_url = 'http://okeanos-master:9870'
hdfs_directory = '/data/ravdess'
#file_list = list_hdfs_files(hdfs_url, hdfs_directory)
#print(file_list)


RAVD = '/data/ravdess/audio_speech_actors_01-24/'
dirl_list = list_hdfs_files(hdfs_url, RAVD)
dirl_list.sort()
print(dirl_list)

emotion = []
gender = []
path = []
for i in dirl_list:
    fname = list_hdfs_files(hdfs_url, RAVD + i)
    for f in fname:
        part = f.split('.')[0].split('-')
        emotion.append(int(part[2]))
        temp = int(part[6])
        if temp%2 == 0:
            temp = "female"
        else:
            temp = "male"
        gender.append(temp)
        path.append(RAVD + i + '/' + f)


print(emotion[:3])
print(gender[:3])
print(path[:3])

hdfs_fs = fs.HadoopFileSystem.from_uri("hdfs://okeanos-master:54310")
#wav = hdfs_fs.open_input_file('/data/ravdess/audio_speech_actors_01-24/Actor_01/03-01-01-01-01-01-01.wav')
#print(librosa.load(wav))

RAVD_df = pd.DataFrame(emotion)
RAVD_df = RAVD_df.replace({1:'neutral', 2:'neutral', 3:'happy', 4:'sad', 5:'angry', 6:'fear', 7:'disgust', 8:'surprise'})
RAVD_df = pd.concat([pd.DataFrame(gender),RAVD_df],axis=1)
RAVD_df.columns = ['gender','emotion']
RAVD_df['labels'] =RAVD_df.gender + '_' + RAVD_df.emotion
RAVD_df['source'] = 'RAVDESS'
RAVD_df = pd.concat([RAVD_df,pd.DataFrame(path, columns = ['path'])],axis=1)
RAVD_df = RAVD_df.drop(['gender', 'emotion'], axis=1)
RAVD_df.labels.value_counts()


# NOISE
def noise(data):
    noise_amp = 0.035*np.random.uniform()*np.amax(data)
    data = data + noise_amp*np.random.normal(size=data.shape[0])
    return data
# STRETCH
def stretch(data, rate=0.8):
    return librosa.effects.time_stretch(data, rate=rate)
# SHIFT
def shift(data):
    shift_range = int(np.random.uniform(low=-5, high = 5)*1000)
    return np.roll(data, shift_range)
# PITCH
def pitch(data, sampling_rate, pitch_factor=0.7):
    return librosa.effects.pitch_shift(data, sr=sampling_rate, n_steps=pitch_factor)

def feat_ext(data, sample_rate):
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    return mfcc

def get_feat(path):
    wavfile = hdfs_fs.open_input_file(path)
    data, sample_rate = librosa.load(wavfile, duration=2.5, offset=0.6)
    # normal data
    res1 = feat_ext(data, sample_rate)
    result = np.array(res1)
    #data with noise
    noise_data = noise(data)
    res2 = feat_ext(noise_data, sample_rate)
    result = np.vstack((result, res2))
    #data with stretch and pitch
    new_data = stretch(data)
    data_stretch_pitch = pitch(new_data, sample_rate)
    res3 = feat_ext(data_stretch_pitch, sample_rate)
    result = np.vstack((result, res3))
    result = np.mean(result, axis=0)
    return result.tolist()


import pyspark
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
import librosa
import numpy as np
from pyspark.sql.functions import col
import os
# create a spark session
#spark = pyspark.sql.SparkSession.builder.appName("AudioFeatureExtraction").getOrCreate()
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

# Initialize Spark
spark = SparkSession \
    .builder \
    .appName("load_data") \
    .master("yarn") \
    .config("spark.executor.instances", sys.argv[1]) \
    .config("spark.executor.cores", "4") \
    .config("spark.jars.packages", "ch.cern.sparkmeasure:spark-measure_2.12:0.23") \
    .getOrCreate()

start_time = time.time()

# create a list of tuples with path and emotion
data = [(path, emotion) for path, emotion in zip(RAVD_df['path'], RAVD_df['labels'])]

# create a spark dataframe from the list
df = spark.createDataFrame(data, ["path", "emotion"])

df.show()

# define a user-defined function to apply the get_feat function on the path column
udf_get_feat = pyspark.sql.functions.udf(get_feat, pyspark.sql.types.ArrayType(pyspark.sql.types.DoubleType()))

# create a new column with the extracted features
df = df.withColumn("features", udf_get_feat(df.path))

# break the features array into 20 columns
df = df.select("emotion", *[df["features"][i] for i in range(20)])

# show the dataframe and write in to csv
df.show()
df.collect()

total_time = time.time() - start_time

print("Time:", total_time)
