# Miniproject: Clustering

Clustering is an unsupervised machine learning task, that is: there are no predefined groups and class labels, but the task is to identify groups so that similar instances will belong to the same group whereas different instances will belong to different groups. 
(Clustering is substantially different from classification: in case of classification, the groups are defined in advance, and the task is to find a model that is able to decide to which group an instance belongs.)

Please note that this page is not a detailed description of clustering algorithms or clustering theory.
In order to understand clustering algorithm you may check out, for example, the textbook "Introduction to Data Mining" (Second Edition) written by Tan, Steinbach, Karpatne and Kumar 
or the manuscript "[Adatbányászat](https://www.cs.bme.hu/~buza/pdfs/adatbanyaszat-cover.pdf)" (in Hungarian).

## k-Means Clustering

One of the most popular clustering algorithms is k-Means. Essentially, we alternate between to steps: 
- assignment of each instance to the closest cluster center, called centroid, and
- selection of new centroids based on the current clustering.
These steps are shown in the code box below:

```
num_iterations = 10
centroids = initial_centroids(data)
for i in range(num_iterations):
  clusters_of_instances = calculate_clusters_of_instances(data, centroids)
  centroids = calculate_new_centroids(data, clusters_of_instances)
```

(If you have learned about expectation-maximizaton (EM) algorithms, you have probably realized by now that k-Mean is one of the EM-algorithms.)

In order to run the above code (and to illustrate how it works), we will import the necessary libaries:

```
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

from google.colab import widgets
from typing import Tuple

from sklearn.metrics.pairwise import euclidean_distances
```


... load a dataset:

```
data_in_frame = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
                   header=None, names=['sepal length', 'sepal width', 'petal length', 'petal width', 'class'])

data = np.zeros( (len(data_in_frame), 4 ))
data[:,0] = data_in_frame['sepal length']
data[:,1] = data_in_frame['sepal width']
data[:,2] = data_in_frame['petal length']
data[:,3] = data_in_frame['petal width']
```

... and define the functions that are referenced:

```
# In fact, the initial centroids are NOT fixed in case of k-Means,
# but we fix them in this example, so that we can study how the result
# changes with different selection of initial centroids
def initial_centroids(data, num_clusters):
  return data[(60,70,80), :]

def calculate_clusters_of_instances(data, centroids):
  return np.argmin(euclidean_distances( data, centroids ), axis=1)

def calculate_new_centroids(data, clusters_of_instances, num_clusters):
  new_centroids = np.zeros( (num_clusters, len(data[0])) )
  for c in range(num_clusters):
    new_centroids[c,:] = np.mean(data[clusters_of_instances==c], axis=0)
  return new_centroids
```

In order to be able to see how the clusters change over time during the iterative process, we will extend to code of k-means so that it saves the clusters of the instances after each iteration:

```
num_clusters = 3
num_iterations = 10
clusters_of_instances_per_iteration = []
centroids = initial_centroids(data, num_clusters)
for i in range(num_iterations):
  clusters_of_instances = calculate_clusters_of_instances(data, centroids)
  centroids = calculate_new_centroids(data, clusters_of_instances, num_clusters)
  clusters_of_instances_per_iteration.append(clusters_of_instances)
```

Next, we will plot what happens in each iteration, we will color each instance according to which cluster it belongs to:

```
def get_color(c):
  if c==0:
    return 'r'
  if c==1:
    return 'b'
  return 'y'

def get_all_colors(clusters_of_instances):
  return [get_color(c) for c in clusters_of_instances]

colors_per_iteration = []
for i in range(num_iterations):
  colors_per_iteration.append(get_all_colors(clusters_of_instances_per_iteration[i]))

tb = widgets.TabBar(list(range(10)), location='top')

for i in range(num_iterations):
  with tb.output_to(i):
    plt.scatter(data[:,2], data[:,3], c=colors_per_iteration[i], marker='x')
    plt.show()
```

If everything went well, the output will look like this (on the tab bar, you can select which iteration you want to see):

![image](https://github.com/user-attachments/assets/e5541af8-b12a-4d12-8c9d-882534848941)

Now you can change the initial centroids and see how the result changes.

Of course, k-Means is implemented in standard libraries as well:

```
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=0).fit(data)
print(kmeans.labels_)
print(kmeans.cluster_centers_)
```

We can plot the clusters at the end of the process as follows: 

```
colors = [get_color(kmeans.labels_[j]) for j in range(len(data))]
p = data_in_frame.plot.scatter('petal length', 'petal width',
                                c = colors, marker='x')
p.set_facecolor('lightgrey')
```

![image](https://github.com/user-attachments/assets/c52e7dc1-6b3a-4f26-a429-f85d5de7a25d)

## Hierarchical clustering

```
from sklearn.cluster import AgglomerativeClustering

model = AgglomerativeClustering(n_clusters=3, linkage='single')
model = model.fit(data)

colors = [get_color(model.labels_[j]) for j in range(len(data))]
p = data_in_frame.plot.scatter('petal length', 'petal width',
                                      c = colors, marker='x')
p.set_facecolor('lightgrey')
```

Let us plot the dendogram showing how hierachical clustering works:

```
from scipy.cluster.hierarchy import dendrogram

def plot_dendrogram(model, **kwargs):
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

# setting distance_threshold=0 ensures we compute the full tree.
model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
model = model.fit(data)

plt.title('Hierarchical Clustering Dendrogram')
# plot the top three levels of the dendrogram
plot_dendrogram(model, truncate_mode='level', p=3)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()
```

![image](https://github.com/user-attachments/assets/3d1c8866-ba03-487a-aac1-255798791fb7)

