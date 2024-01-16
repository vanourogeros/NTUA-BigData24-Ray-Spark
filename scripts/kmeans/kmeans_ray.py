import sys
import ray
from ray.data.preprocessors import Concatenator
from typing import List
from ray.data.aggregate import AggregateFn

import numpy as np

def closestPoint(row, centers: List[np.ndarray]):
    bestIndex = 0
    p = row['features']
    closest = float("+inf")
    for i in range(len(centers)):
        tempDist = np.sum((p - centers[i]) ** 2)
        if tempDist < closest:
            closest = tempDist
            bestIndex = i
    row['chosen_class'] = bestIndex
    return row

ds = ray.data.read_csv("/content/mixed_dataset.csv") \
              .select_columns(['feature_1', 'feature_2', 'feature_3'])

preproccessor = Concatenator(output_column_name="features")
ds = preproccessor.fit_transform(ds)
n_points = ds.count()
# Choose two random samples from the dataset
centers = ds.take(2)
centers = [row['features'] for row in centers]

# Print the centers
print(f"The chosen centers are:\n{centers}")

max_iters = 10
for _ in range(max_iters):
  ds = ds.map(lambda row: closestPoint(row, centers))
  #ds.show()

  aggregation = AggregateFn(
      init=lambda column: np.array([0.0,0.0,0.0]),
      accumulate_row=lambda a, row: a+row['features']/n_points,
      merge = lambda a1, a2: a1+a2,
      name="new_centers"
  )

  # Group by 'chosen_class' and take the mean of 'features'
  grouped = ds.groupby('chosen_class').aggregate(aggregation)

  centers = [center["new_centers"] for center in grouped.iter_rows()]

# The result is a pandas Series with 'chosen_class' as the index and the mean 'features' as the values

print(centers)
ds.show()


""" visualization of the classes

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Convert the Ray dataset to a NumPy array
features_list = [item['features'] for item in ds.iter_rows()]
features_array = np.array(features_list)

# Extract classes
classes = np.array([item['chosen_class'] for item in ds.iter_rows()])

print(np.unique(classes))

# Create a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot points with different colors based on the class
ax.scatter(features_array[:, 0], features_array[:, 1], features_array[:, 2], c=classes, cmap='viridis')


plt.title("K-means clustering with Ray")
plt.show()
print(len(features_list))
"""
