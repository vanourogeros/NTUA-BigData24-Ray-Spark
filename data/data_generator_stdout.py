import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from faker import Faker
import sys
import os

"""
NOTE: This script is to be used with redirection of std output
      to some other source, e.g. an hdfs with
      `python generate_data_stdout.py 10000 | hdfs dfs -put - /generated_data.csv`
"""

# Create a Faker instance
faker = Faker()

def generate_mixed_data_chunk(chunk_size):
    # Generate synthetic data with numeric features
    numeric_features, labels = make_classification(
        n_samples=chunk_size,
        n_features=3,
        n_informative=3,
        n_redundant=0,
        random_state=42
    )

    # Convert numeric features to a pandas DataFrame
    numeric_df = pd.DataFrame(data=numeric_features, columns=[f'feature_{i+1}' for i in range(3)])

    # Generate synthetic categorical features (strings)
    categorical_df = pd.DataFrame({
        'categorical_feature_1': np.random.randint(1, chunk_size, size=chunk_size),
        'categorical_feature_2': np.random.randint(1, chunk_size//2, size=chunk_size),
        'word': [faker.word() for _ in range(chunk_size)]
    })

    # Combine numeric and categorical features
    mixed_df = pd.concat([numeric_df, categorical_df], axis=1)

    # Combine features and labels
    df_chunk = pd.concat([mixed_df, pd.DataFrame({'label': labels})], axis=1)

    return df_chunk

def create_mixed_large_csv(num_samples, chunk_size=1000):
        # Create header to the file
        header = [f'feature_{i+1}' for i in range(3)] + ['categorical_feature_1', 'categorical_feature_2','word', 'label']
        #f.write(','.join(header) + '\n')

        # Write header to the standard output
        print(','.join(header))
        # Generate and write data in chunks
        for _ in range(0, num_samples, chunk_size):
            data_chunk = generate_mixed_data_chunk(chunk_size)
            data_chunk.to_csv(sys.stdout, header=False, index=False, mode='a')

# Example usage
num_samples = int(sys.argv[1]) # should be a multiple of chunk_size
create_mixed_large_csv(num_samples=num_samples)
